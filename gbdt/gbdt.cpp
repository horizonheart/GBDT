#include <limits>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>

#include "gbdt.h"
#include "timer.h"

/*
This implementation is based on Algo 5 in the following paper:

   http://statweb.stanford.edu/~jhf/ftp/trebst.pdf

We do not have a plan to maintain this part of code, so there is no documents.

We recommend you either check the above document, or use some exisiting packages such as xgboost.
*/

namespace {
	//计算F0(x)
float calc_bias(std::vector<float> const &Y)
{
    double y_bar = std::accumulate(Y.begin(), Y.end(), 0.0); //累加所有的label值
    y_bar /= static_cast<double>(Y.size());
    return static_cast<float>(log((1.0+y_bar)/(1.0-y_bar)));
}
//记录每一个样本的信息
//为每条样本记录树中的位置，以及残差，是否收缩
//shrinkage作用：不完全信任每一棵树，累加的时候以一定的权重只累加一部分
struct Location
{
    Location() : tnode_idx(1), r(0), shrinked(false) {}
    uint32_t tnode_idx; //样本所属的节点的索引
    float r;//当前样本的残差
    bool shrinked;
};
//记录每一个叶子节点的信息
struct Meta
{
    Meta() : sl(0), s(0), nl(0), n(0), v(0.0f/1.0f) {}
    double sl, s;//分别代表左子树和当前节点残差的总和
    uint32_t nl, n; //分别代表左子树和当前节点的样本的数目
    float v;//当前叶子节点所输出的值
};
//叶节点的每一维特征
//应该是分裂出的节点，ese是分裂值
struct Defender
{
    Defender() : ese(0), threshold(0) {}
    double ese;
    float threshold;//记录cart树划分的标准，按照哪个特征进行划分
};

//************************************
// Method:    scan
// FullName:  scan
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: Problem const & prob 
// Parameter: std::vector<Location> const & locations 样本的信息
// Parameter: std::vector<Meta> const & metas0 每个叶子节点的信息
// Parameter: std::vector<Defender> & defenders 每个叶子每个特征的信息
// Parameter: uint32_t const offset
// Parameter: bool const forward
//************************************
void scan(
    Problem const &prob,
    std::vector<Location> const &locations,
    std::vector<Meta> const &metas0,
    std::vector<Defender> &defenders,
    uint32_t const offset,
    bool const forward)
{
    uint32_t const nr_field = prob.nr_field;//特征的数目
    uint32_t const nr_instance = prob.nr_instance;//样本的个数

    #pragma omp parallel for schedule(dynamic)
    for(uint32_t j = 0; j < nr_field; ++j) //依次循环每一个特征
    {
        std::vector<Meta> metas = metas0; //叶子节点的信息

        for(uint32_t i_bar = 0; i_bar < nr_instance; ++i_bar)
        {
            uint32_t const i = forward? i_bar : nr_instance-i_bar-1;

            Node const &dnode = prob.X[j][i]; //获取第j个特征的第i行的信息
            Location const &location = locations[dnode.i];//第i个样本的信息
            if(location.shrinked)
                continue;

            uint32_t const f = location.tnode_idx-offset; //找到样本所在的叶子节点的索引
            Meta &meta = metas[f];//对应的叶子节点

            if(dnode.v != meta.v)//如果当前特征的值和叶子节点的输出值不一样，此时就要将改节点继续进行划分
            {
                double const sr = meta.s - meta.sl;
                uint32_t const nr = meta.n - meta.nl;
                double const current_ese = 
                    (meta.sl*meta.sl)/static_cast<double>(meta.nl) + 
                    (sr*sr)/static_cast<double>(nr);//计算如果按照当前特征划分的话，残差的和为多少

                Defender &defender = defenders[f*nr_field+j];
                double &best_ese = defender.ese;
                if(current_ese > best_ese)
                {
                    best_ese = current_ese;
                    defender.threshold = forward? dnode.v : meta.v;
                }
                if(i_bar > nr_instance/2) //如果当前节点的数目
                    break;
            }

            meta.sl += location.r;
            ++meta.nl;
            meta.v = dnode.v;
        }
    }
}

void scan_sparse(
    Problem const &prob,
    std::vector<Location> const &locations,
    std::vector<Meta> const &metas0,
    std::vector<Defender> &defenders,
    uint32_t const offset,
    bool const forward)
{
    uint32_t const nr_sparse_field = prob.nr_sparse_field;
    uint32_t const nr_leaf = offset;

    #pragma omp parallel for schedule(dynamic)
    for(uint32_t j = 0; j < nr_sparse_field; ++j)
    {
        std::vector<Meta> metas = metas0;
        for(uint64_t p = prob.SIP[j]; p < prob.SIP[j+1]; ++p)
        {
            Location const &location = locations[prob.SI[p]];
            if(location.shrinked)
                continue;
            Meta &meta = metas[location.tnode_idx-offset];
            meta.sl += location.r;
            ++meta.nl;
        }

        for(uint32_t f = 0; f < nr_leaf; ++f)
        {
            Meta const &meta = metas[f];
            if(meta.nl == 0)
                continue;
            
            double const sr = meta.s - meta.sl;
            uint32_t const nr = meta.n - meta.nl;
            double const current_ese = 
                (meta.sl*meta.sl)/static_cast<double>(meta.nl) + 
                (sr*sr)/static_cast<double>(nr);

            Defender &defender = defenders[f*nr_sparse_field+j];
            double &best_ese = defender.ese;
            if(current_ese > best_ese)
            {
                best_ese = current_ese;
                defender.threshold = 1;
            }
        }
    }
}

} //unnamed namespace

uint32_t CART::max_depth = 7;
uint32_t CART::max_tnodes = static_cast<uint32_t>(pow(2, CART::max_depth+1));
std::mutex CART::mtx;
bool CART::verbose = false;

//************************************
// Method:    fit
// FullName:  CART::fit
// Access:    public 
// Returns:   void
// Qualifier: 根据残差训练CART树
// Parameter: Problem const & prob
// Parameter: std::vector<float> const & R 残差，负梯度方向
// Parameter: std::vector<float> & F1
//************************************
void CART::fit(Problem const &prob, std::vector<float> const &R, 
    std::vector<float> &F1)
{
    uint32_t const nr_field = prob.nr_field;// 特征的个数
    uint32_t const nr_sparse_field = prob.nr_sparse_field;
    uint32_t const nr_instance = prob.nr_instance;// 样本的个数

    std::vector<Location> locations(nr_instance);  // 样本信息
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
        locations[i].r = R[i]; // 记录每一个样本的残差
    for(uint32_t d = 0, offset = 1; d < max_depth; ++d, offset *= 2) // d:深度
    {
        uint32_t const nr_leaf = static_cast<uint32_t>(pow(2, d));// 叶子节点的个数
        std::vector<Meta> metas0(nr_leaf); // 叶子节点的信息

		//计算所有总的残差
        for(uint32_t i = 0; i < nr_instance; ++i)
        {
            Location &location = locations[i]; //第i个样本的信息
            if(location.shrinked)
                continue;

			Meta &meta = metas0[location.tnode_idx - offset]; //找到对应的叶子节点
            meta.s += location.r;//残差之和
            ++meta.n;
        }

		std::vector<Defender> defenders(nr_leaf*nr_field); //记录每一个叶节点的每一维特征
        std::vector<Defender> defenders_sparse(nr_leaf*nr_sparse_field);

        for(uint32_t f = 0; f < nr_leaf; ++f)
        {
            Meta const &meta = metas0[f];//拿到当前的叶子节点
            double const ese = meta.s*meta.s/static_cast<double>(meta.n);//计算当前叶子节点的ese
            for(uint32_t j = 0; j < nr_field; ++j)
                defenders[f*nr_field+j].ese = ese;
            for(uint32_t j = 0; j < nr_sparse_field; ++j)
                defenders_sparse[f*nr_sparse_field+j].ese = ese;
        }
        std::vector<Defender> defenders_inv = defenders;

        std::thread thread_f(scan, std::ref(prob), std::ref(locations),
            std::ref(metas0), std::ref(defenders), offset, true);
        std::thread thread_b(scan, std::ref(prob), std::ref(locations),
            std::ref(metas0), std::ref(defenders_inv), offset, false);
        scan_sparse(prob, locations, metas0, defenders_sparse, offset, true);
        thread_f.join();
        thread_b.join();
		// 找出最佳的ese，scan里是每个字段的最佳ese，这里是所有字段的最佳ese，赋值给相应的tnode
        for(uint32_t f = 0; f < nr_leaf; ++f)
        {
            Meta const &meta = metas0[f];
            double best_ese = meta.s*meta.s/static_cast<double>(meta.n);
            TreeNode &tnode = tnodes[f+offset];
            for(uint32_t j = 0; j < nr_field; ++j)
            {
                Defender defender = defenders[f*nr_field+j];
                if(defender.ese > best_ese)
                {
                    best_ese = defender.ese;
                    tnode.feature = j;
                    tnode.threshold = defender.threshold;
                }

                defender = defenders_inv[f*nr_field+j];
                if(defender.ese > best_ese)
                {
                    best_ese = defender.ese;
                    tnode.feature = j;
                    tnode.threshold = defender.threshold;
                }
            }
            for(uint32_t j = 0; j < nr_sparse_field; ++j)
            {
                Defender defender = defenders_sparse[f*nr_sparse_field+j];
                if(defender.ese > best_ese)
                {
                    best_ese = defender.ese;
                    tnode.feature = nr_field + j;
                    tnode.threshold = defender.threshold;
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < nr_instance; ++i)
        {
            Location &location = locations[i];
            if(location.shrinked)
                continue;

            uint32_t &tnode_idx = location.tnode_idx;
            TreeNode &tnode = tnodes[tnode_idx];
            if(tnode.feature == -1)
            {
                location.shrinked = true;
            }
            else if(static_cast<uint32_t>(tnode.feature) < nr_field)
            {
                if(prob.Z[tnode.feature][i].v < tnode.threshold)
                    tnode_idx = 2*tnode_idx; 
                else
                    tnode_idx = 2*tnode_idx+1; 
            }
            else
            {
                uint32_t const target_feature 
                    = static_cast<uint32_t>(tnode.feature-nr_field);
                bool is_one = false;
                for(uint64_t p = prob.SJP[i]; p < prob.SJP[i+1]; ++p) 
                {
                    if(prob.SJ[p] == target_feature)
                    {
                        is_one = true;
                        break;
                    }
                }
                if(!is_one)
                    tnode_idx = 2*tnode_idx; 
                else
                    tnode_idx = 2*tnode_idx+1; 
            }
        }
    }

    std::vector<std::pair<double, double>> 
        tmp(max_tnodes, std::make_pair(0, 0));
    for(uint32_t i = 0; i < nr_instance; ++i)
    {
        float const r = locations[i].r;
        uint32_t const tnode_idx = locations[i].tnode_idx;
        tmp[tnode_idx].first += r;
        tmp[tnode_idx].second += fabs(r)*(1-fabs(r));
    }

    for(uint32_t tnode_idx = 1; tnode_idx <= max_tnodes; ++tnode_idx)
    {
        double a, b;
        std::tie(a, b) = tmp[tnode_idx];
        tnodes[tnode_idx].gamma = (b <= 1e-12)? 0 : static_cast<float>(a/b);
    }

    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
        F1[i] = tnodes[locations[i].tnode_idx].gamma;
}

std::pair<uint32_t, float> CART::predict(float const * const x) const
{
    uint32_t tnode_idx = 1;
    for(uint32_t d = 0; d <= max_depth; ++d)
    {
        TreeNode const &tnode = tnodes[tnode_idx];
        if(tnode.feature == -1)
            return std::make_pair(tnode.idx, tnode.gamma);

        if(x[tnode.feature] < tnode.threshold)
            tnode_idx = tnode_idx*2;
        else
            tnode_idx = tnode_idx*2+1;
    }

    return std::make_pair(-1, -1);
}

void GBDT::fit(Problem const &Tr, Problem const &Va)
{
    bias = calc_bias(Tr.Y);//计算初始值F0

    std::vector<float> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    Timer timer;
    printf("iter     time    tr_loss    va_loss\n");
	// 开始训练每一棵CART树
    for(uint32_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<float> const &Y = Tr.Y;
        std::vector<float> R(Tr.nr_instance), F1(Tr.nr_instance);// 记录残差和F(生成树)

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
            R[i] = static_cast<float>(Y[i]/(1+exp(Y[i]*F_Tr[i])));//计算残差，或者称为梯度下降的方向
		// 利用上面的残差值，在此函数中构造一棵树
        trees[t].fit(Tr, R, F1); // 分类树的生成

        double Tr_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Tr_loss)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += F1[i];
            Tr_loss += log(1+exp(-Y[i]*F_Tr[i]));
        }
        Tr_loss /= static_cast<double>(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Va.nr_instance; ++i)
        {
            std::vector<float> x = construct_instance(Va, i);
            F_Va[i] += trees[t].predict(x.data()).second;
        }

        double Va_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Va_loss)
        for(uint32_t i = 0; i < Va.nr_instance; ++i) 
            Va_loss += log(1+exp(-Va.Y[i]*F_Va[i]));
        Va_loss /= static_cast<double>(Va.nr_instance);

        printf("%4d %8.1f %10.5f %10.5f\n", t, timer.toc(), Tr_loss, Va_loss);
        fflush(stdout);
    }
}

float GBDT::predict(float const * const x) const
{
    float s = bias;
    for(auto &tree : trees)
        s += tree.predict(x).second;
    return s;
}

std::vector<uint32_t> GBDT::get_indices(float const * const x) const
{
    uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

    std::vector<uint32_t> indices(nr_tree);
    for(uint32_t t = 0; t < nr_tree; ++t)
        indices[t] = trees[t].predict(x).first;
    return indices;
}

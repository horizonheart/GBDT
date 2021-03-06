#define  _CRT_SECURE_NO_WARNINGS
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include "common.h"
#include <stdint.h>
namespace {

 uint32_t const kMaxLineSize = 1000000;
//获取文件的行数
uint32_t get_nr_line(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
	/*
	char *fgets(char *buf, int bufsize, FILE *stream):从文件结构体指针stream中读取数据，每次读取一行。
	读取的数据保存在buf指向的字符数组中，每次最多读取bufsize-1个字符（第bufsize个字符赋'\0'），
	如果文件中的该行不足bufsize个字符，则读完该行就结束。
	*/
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}
//获取特征的数目
uint32_t get_nr_field(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");

    uint32_t nr_field = 0;
    while(1)
    {
        char *val_char = strtok(nullptr," \t");
        if(val_char == nullptr || *val_char == '\n')
            break;
        ++nr_field;
    }

    fclose(f);

    return nr_field;
}
//读取稠密矩阵
void read_dense(Problem &prob, std::string const &path)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *p = strtok(line, " \t");//分割字符串
        prob.Y[i] = (atoi(p)>0)? 1.0f : -1.0f;
        for(uint32_t j = 0; j < prob.nr_field; ++j)
        {
            char *val_char = strtok(nullptr," \t");

            float const val = static_cast<float>(atof(val_char));

            prob.X[j][i] = Node(i, val);
        }
    }

    fclose(f);
}
//排序
void sort_problem(Problem &prob)
{
    struct sort_by_v
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.v < rhs.v;
        }
    };

    #pragma omp parallel for schedule(static)
    for(uint32_t j = 0; j < prob.nr_field; ++j)
    {
        std::vector<Node> &X1 = prob.X[j]; //获取到第一个数值特征所有的数值
        std::vector<Node> &Z1 = prob.Z[j];
        std::sort(X1.begin(), X1.end(), sort_by_v());
        for(uint32_t i = 0; i < prob.nr_instance; ++i)
            Z1[X1[i].i] = Node(i, X1[i].v);
    }
}
//读取稀疏矩阵
void read_sparse(Problem &prob, std::string const &path)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");

    std::vector<std::vector<uint32_t>> buffer;//buffer存储了稀疏，第一维存储的是非零元素的索引，第二维存储的是非零元素所在的行索引，其实就是该非零元素在哪一行存在

    uint64_t nnz = 0; 
    uint32_t nr_instance = 0;
    prob.SJP.push_back(0);
    for(; fgets(line, kMaxLineSize, f) != nullptr; ++nr_instance)
    {
        strtok(line, " \t");
        for( ; ; ++nnz)
        {
            char *idx_char = strtok(nullptr," \t");
            if(idx_char == nullptr || *idx_char == '\n')
                break;

            uint32_t const idx = atoi(idx_char);
            if(idx > buffer.size())
                buffer.resize(idx);
            buffer[idx-1].push_back(nr_instance);
            prob.SJ.push_back(idx-1);//SJ里面了存储了所有的非零元素的索引
        }
        prob.SJP.push_back(prob.SJ.size());//SJP存储了每一行非零元素的个数 0 9	18 25 31 36	
    }
    prob.SJ.shrink_to_fit();//确保capacity和size的大小一样，避免浪费内存
    prob.SJP.shrink_to_fit();

    prob.nr_sparse_field = static_cast<uint32_t>(buffer.size());//计算非空字段的数目
    prob.SI.resize(nnz); //nnz总共有多少个非零的字段
    prob.SIP.resize(prob.nr_sparse_field+1); //这个是按照行先存储，先存储第一个非零字段的所有的行，再存储第二个非空的行
    prob.SIP[0] = 0;

    uint64_t p = 0;
    for(uint32_t j = 0; j < prob.nr_sparse_field; ++j)
    {
        for(auto i : buffer[j]) 
            prob.SI[p++] = i;
        prob.SIP[j+1] = p;
    }

    fclose(f);

    sort_problem(prob);
}

} //unamed namespace
//读取文件数据
Problem read_data(std::string const &dense_path, std::string const &sparse_path)
{
    Problem prob(get_nr_line(dense_path), get_nr_field(dense_path));

    read_dense(prob, dense_path);

    read_sparse(prob, sparse_path);

    return prob;
}
//打开一个文件
FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}
//输入参数转化为vector返回
std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

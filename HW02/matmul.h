#ifndef MATMUL_H
#define MATMUL_H

#include <vector>

void mmul1(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n);
void mmul2(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n);
void mmul3(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n);
void mmul4(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, int n);

#endif // MATMUL_H

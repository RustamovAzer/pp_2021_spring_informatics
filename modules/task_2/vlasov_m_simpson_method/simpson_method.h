// Copyright 2021 Vlasov Maksim

#ifndef MODULES_TASK_2_VLASOV_M_SIMPSON_METHOD_SIMPSON_METHOD_H_
#define MODULES_TASK_2_VLASOV_M_SIMPSON_METHOD_SIMPSON_METHOD_H_

#include <functional>
#include <vector>

namespace SimpsonMethod {

double sequential(const std::function<double(const std::vector<double>&)>& func,
                  const std::vector<double>& seg_begin,
                  const std::vector<double>& seg_end, int steps_count);

double parallel(const std::function<double(const std::vector<double>&)>& func,
                std::vector<double> seg_begin, std::vector<double> seg_end,
                int steps_count);

}  // namespace SimpsonMethod

#endif  // MODULES_TASK_2_VLASOV_M_SIMPSON_METHOD_SIMPSON_METHOD_H_

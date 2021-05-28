// Copyright 2021 Schekotilova Julia
#define NOMINMAX
#include <vector>
#include "../../../modules/task_3/schekotilova_ju_radix_merge_double/radix_merge_double.h"

std::vector<double> sorting(std::vector<double> in, std::vector<double> out, const int val) {
  const int size = in.size();
  unsigned char* arr = (unsigned char*)in.data();
  int counter[256], res = 0;
  for (int i = 0; i < 256; i++) counter[i] = 0;
  for (int i = 0; i < size; i++) counter[arr[8 * i + val]]++;
  for (int i = 0; i < 256; i++) {
    int out = counter[i];
    counter[i] = res;
    res += out;
  }
  for (int i = 0; i < size; i++) {
    out[counter[arr[8 * i + val]]] = in[i];
    counter[arr[8 * i + val]]++;
  }
  return out;
}

std::vector<double> last_sorting(std::vector<double> in) {
  std::vector<double> out = std::vector<double>(in);
  int const val = 7;
  int size = in.size();
  int res = 0, counter[256];
  unsigned char* arr = (unsigned char*)in.data();
  for (int i = 0; i < 256; i++) counter[i] = 0;

  for (int i = 0; i < size; i++) counter[arr[8 * i + val]]++;
  for (int i = 255; i > 127; i--) {
    counter[i] += res;
    res = counter[i];
  }
  for (int i = 0; i < 128; i++) {
    int tmp = counter[i];
    counter[i] = res;
    res += tmp;
  }
  for (int i = 0; i < size; i++) {
    if (arr[8 * i + val] < 128) {
      out[counter[arr[8 * i + val]]] = in[i];
      counter[arr[8 * i + val]]++;
    } else {
      counter[arr[8 * i + val]]--;
      out[counter[arr[8 * i + val]]] = in[i];
    }
  }
  return out;
}

std::vector<double> radix_sort(const std::vector<double>& in) {
  std::vector<double> out = std::vector<double>(in.size()), tmp = std::vector<double>(in.size());
  std::vector<double> arr = std::vector<double>(in);
  std::vector<double> k;
  for (int i = 0; i < 8; i++) {
    out = sorting(arr, out, i);
    tmp = arr;
    arr = out;
    out = tmp;
  }
  k = last_sorting(arr);
  return k;
}

std::vector<double> radix_sort_omp(const std::vector<double>& in) {
  int thrd = 4, size_arr, counter;
  std::vector<double> res;
  std::vector<std::vector<double>> arr, result;
  size_arr = (in.size() - 1) / thrd + 1;
  arr = std::vector<std::vector<double>>((in.size() + size_arr) / size_arr);
  for (size_t i = 0; i < in.size(); i += size_arr) {
    int last = std::min(in.size(), (i + size_arr));
    int idx = i / size_arr;
    std::vector<double>& vector = arr[idx];
    vector.reserve(last - i);
    move(in.begin() + i, in.begin() + last, back_inserter(vector));
  }
  counter = arr.size();
#pragma omp parallel num_threads(thrd)
#pragma omp for
  for (int i = 0; i < counter; i++) {
    std::vector<double>  res_arr = radix_sort(arr[i]);
#pragma omp critical
    result.push_back(res_arr);
  }
  res = result[0];
  if (counter != 1) {
    for (int i = 0; i < counter - 1; i++) {
      res = merge(res, result[i + 1]);
    }
  }
  return res;
}

std::vector<double> radix_sort_tbb(const std::vector<double>& in) {
  int thrd = 4, size_arr, counter;
  std::vector<double> res;
  std::vector<std::vector<double>> arr, result;
  size_arr = (in.size() - 1) / thrd + 1;
  arr = std::vector<std::vector<double>>((in.size() + size_arr) / size_arr);
  for (size_t i = 0; i < in.size(); i += size_arr) {
    int last = std::min(in.size(), (i + size_arr));
    int idx = i / size_arr;
    std::vector<double>& vector = arr[idx];
    vector.reserve(last - i);
    move(in.begin() + i, in.begin() + last, back_inserter(vector));
  }
  counter = arr.size();

  tbb::task_scheduler_init init(thrd);
  tbb::parallel_for(0, counter, [&](const int i) {
    arr[i] = radix_sort(arr[i]);
  });
  res = arr[0];
  for (int i = 0; i < counter - 1; i++) {
    res = merge(res, arr[i + 1]);
  }
  return res;
}

  std::vector<double> merge(const std::vector<double> & arr, const std::vector<double> & arr_) {
  const int size = arr.size(), size_ = arr_.size();
  int i = 0, j = 0, k = 0;
  std::vector<double> result(size + size_);
  for (k = 0; (k < (size + size_ - 1)) && ((i < size) && (j < size_)); k++) {
    if (arr[i] < arr_[j]) {
      result[k] = arr[i++];
    } else {
      result[k] = arr_[j++];
    }
  }
  while (i < size) {
    result[k] = arr[i];
    k++;
    i++;
  }
  while (j < size_) {
    result[k] = arr_[j];
    k++;
    j++;
  }
  return result;
}

bool checker(std::vector<double> arr, int size) {
  for (int i = 0; i < size - 1; i++) {
    if (arr[i + 1] >= arr[i]) {
      i++;
    } else {
      return false;
    }
  }
  return true;
}

std::vector<double> generate(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

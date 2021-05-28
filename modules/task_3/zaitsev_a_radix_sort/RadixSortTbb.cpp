// Copyright 2021 Zaitsev Andrey
#include <tbb/tbb.h>
#include <random>
#include <ctime>
#include <utility>
#include <vector>
#include <queue>
#include <stack>
#include "../../../modules/task_3/zaitsev_a_radix_sort/RadixSortTbb.h"

std::vector<double> createVector(int size, double a, double b) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::uniform_real_distribution<double> distr(a, b);
  std::vector<double> vec;
  for (int i = 0; i < size; i++) {
    vec.push_back(distr(gen));
  }
  return vec;
}

bool check(double* vec, int size) {
  for (int i = 0; i < size - 1; i++) {
    if (vec[i + 1] >= vec[i]) {
      i++;
    } else {
      return false;
    }
  }
  return true;
}

void radixSort(double *vec, double *out, int size) {
  double* swap = nullptr;
  for (int i = 0; i < 7; i++) {
    unsigned char* temp_vec_one_byte = (unsigned char*)vec;
    int cnt[256] = { 0 };
    int res = 0;
    for (int j = 0; j < size; j++)
      cnt[temp_vec_one_byte[8 * j + i]]++;
    for (int j = 0; j < 256; j++) {
      int tmp_ = cnt[j];
      cnt[j] = res;
      res += tmp_;
    }
    for (int j = 0; j < size; j++) {
      int ind = 8 * j + i;
      out[cnt[temp_vec_one_byte[ind]]] = vec[j];
      cnt[temp_vec_one_byte[ind]]++;
    }

    swap = vec;
    vec = out;
    out = swap;
  }

  unsigned char* temp_vec_one_byte = (unsigned char*)vec;
  int cnt[256] = { 0 };
  int shift[256] = { 0 };
  for (int i = 0; i < size; i++)
    cnt[temp_vec_one_byte[8 * i + 7]]++;
  shift[255] = cnt[255] - 1;
  for (int i = 254; i >= 128; i--)
    shift[i] = shift[i + 1] + cnt[i];
  shift[0] = shift[128] + 1;
  for (int i = 1; i < 128; i++)
    shift[i] = shift[i - 1] + cnt[i - 1];
  for (int i = 0; i < size; i++) {
    if (temp_vec_one_byte[8 * i + 7] < 128)
      out[shift[temp_vec_one_byte[8 * i + 7]]++] = vec[i];
    else
      out[shift[temp_vec_one_byte[8 * i + 7]]--] = vec[i];
  }
}

void merge(double *vec, double *out, int center, int size, int odd_even) {
  for (int i = odd_even; i < center; i += 2)
    out[i] = vec[i];
  double *vec_from_center = vec + center;
  int par1 = odd_even;
  int par2 = odd_even;
  int ind = odd_even;
  while ((par1 < center) && (par2 < size)) {
    if (out[par1] > vec_from_center[par2]) {
      vec[ind] = vec_from_center[par2];
      par2 += 2;
    } else {
      vec[ind] = out[par1];
      par1 += 2;
    }
    ind += 2;
  }
  if (par1 != center) {
    for (int j = par1; j < center; j += 2, ind += 2)
      vec[ind] = out[j];
  } else {
    for (int j = par2; j < size; j += 2, ind += 2)
      vec[ind] = vec_from_center[j];
  }
}

void oddEvenMergeSortSeq(double* vec, int size, int border) {
  double* temp_vec = new double[size];
  std::vector<std::pair<int, int>> piecesOfVector;
  std::stack<std::pair<int, std::pair<int, int>>> preparingForMerges;
  std::queue<std::pair<int, int>> curr_level;
  std::queue<std::pair<int, int>> next_level;

  preparingForMerges.push({ 0, {0, 0} });
  curr_level.push({ 0, size });

  std::pair<int, int> current;

  while (true) {
    current = curr_level.front();
    curr_level.pop();
    if (current.second > border) {
      int center = current.second / 2 + (current.second / 2) % 2;
      next_level.push({ current.first, center });
      next_level.push({ current.first + center, current.second - center });
      preparingForMerges.push({ current.first,
        {center, current.second - center} });
    } else {
      piecesOfVector.push_back(current);
    }
    if (curr_level.empty()) {
      if (!next_level.empty()) {
        curr_level = next_level;
        next_level = std::queue<std::pair<int, int>>();
        preparingForMerges.push({ 0, {0, 0} });
      } else {
        break;
      }
    }
  }
  preparingForMerges.pop();

  for (size_t i = 0; i < piecesOfVector.size(); i++) {
    radixSort(vec + piecesOfVector[i].first,
      temp_vec + piecesOfVector[i].first, piecesOfVector[i].second);
  }
  std::vector<std::pair<int, std::pair<int, int>>> merges_order;

  while (!preparingForMerges.empty()) {
    while (preparingForMerges.top().second.second != 0) {
      merges_order.push_back(preparingForMerges.top());
      preparingForMerges.pop();
    }
    preparingForMerges.pop();

    for (size_t i = 0; i < merges_order.size() * 2; i++) {
      int start = merges_order[i / 2].first;
      int center = merges_order[i / 2].second.first;
      int size_after_center = merges_order[i / 2].second.second;
      merge(vec + start, temp_vec + start, center, size_after_center, i % 2);
    }

    for (size_t i = 0; i < merges_order.size(); i++) {
      int start = merges_order[i].first;
      int center = merges_order[i].second.first;
      int size_after_center = merges_order[i].second.second;
      for (int j = 1; j < (center + size_after_center + 1) / 2; j++) {
        if (vec[start + 2 * j - 1] > vec[start + 2 * j]) {
          double tmp = vec[start + 2 * j];
          vec[start + 2 * j] = vec[start + 2 * j - 1];
          vec[start + 2 * j - 1] = tmp;
        }
      }
    }
    merges_order.clear();
  }
  delete[] temp_vec;
}

void oddEvenMergeSortTbb(double* vec, int size, int border, int threads) {
  double* temp_vec = new double[size];
  std::vector<std::pair<int, int>> piecesOfVector;
  std::stack<std::pair<int, std::pair<int, int>>> preparingForMerges;
  std::queue<std::pair<int, int>> curr_level;
  std::queue<std::pair<int, int>> next_level;

  preparingForMerges.push({ 0, {0, 0} });
  curr_level.push({ 0, size });

  std::pair<int, int> current;

  while (true) {
    current = curr_level.front();
    curr_level.pop();
    if (current.second > border) {
      int center = current.second / 2 + (current.second / 2) % 2;
      next_level.push({ current.first, center });
      next_level.push({ current.first + center, current.second - center });
      preparingForMerges.push({ current.first,
        {center, current.second - center} });
    } else {
      piecesOfVector.push_back(current);
    }
    if (curr_level.empty()) {
      if (!next_level.empty()) {
        curr_level = next_level;
        next_level = std::queue<std::pair<int, int>>();
        preparingForMerges.push({ 0, {0, 0} });
      } else {
        break;
      }
    }
  }
  preparingForMerges.pop();

  int n = static_cast<int>(piecesOfVector.size());
  tbb::task_scheduler_init init(threads);
  tbb::parallel_for(tbb::blocked_range<int>(0, n, 1),
    [&](const tbb::blocked_range<int>& range) {
      int begin = range.begin();
      int end = range.end();
      for (int i = begin; i < end; i++) {
          radixSort(vec + piecesOfVector[i].first,
              temp_vec + piecesOfVector[i].first, piecesOfVector[i].second);
      }
    });
  init.terminate();

  std::vector<std::pair<int, std::pair<int, int>>> merges_order;

  while (!preparingForMerges.empty()) {
    while (preparingForMerges.top().second.second != 0) {
      merges_order.push_back(preparingForMerges.top());
      preparingForMerges.pop();
    }
    preparingForMerges.pop();

    n = static_cast<int>(merges_order.size() * 2);
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1),
      [&](const tbb::blocked_range<int>& range) {
        int begin = range.begin();
        int end = range.end();
        for (int i = begin; i < end; i++) {
          int start = merges_order[i / 2].first;
          int center = merges_order[i / 2].second.first;
          int size_after_center = merges_order[i / 2].second.second;
          merge(vec + start, temp_vec + start,
            center, size_after_center, i % 2);
        }
      });

    n = static_cast<int>(merges_order.size());
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1),
      [&](const tbb::blocked_range<int>& range) {
        int begin = range.begin();
        int end = range.end();
        for (int i = begin; i < end; i++) {
          int start = merges_order[i].first;
          int center = merges_order[i].second.first;
          int size_after_center = merges_order[i].second.second;
          for (int j = 1; j < (center + size_after_center + 1) / 2; j++) {
            if (vec[start + 2 * j - 1] > vec[start + 2 * j]) {
              double tmp = vec[start + 2 * j];
              vec[start + 2 * j] = vec[start + 2 * j - 1];
              vec[start + 2 * j - 1] = tmp;
            }
          }
        }
      });
    merges_order.clear();
  }
  delete[] temp_vec;
}

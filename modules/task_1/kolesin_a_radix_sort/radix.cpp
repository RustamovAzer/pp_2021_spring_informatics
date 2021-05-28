// Copyright 2021 Kolesin Andrey

#include <iostream>
#include "../../../modules/task_1/kolesin_a_radix_sort/radix.h"
const int base = 256;
const int shift = 8;
const int digits = 4;
const unsigned int masks[digits] = {255u, 255u << 8, 255u << 16, 255u << 24};

void printVec(std::vector<int> vec) {
  if (vec.empty()) {
    return;
  }
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

unsigned int getDigit(int num, int p) {
  unsigned int res = num & masks[p];
  return res >> (shift * p);
}

void radix(int* arr, int size, int p) {
  std::vector<int> digit[base];
  for (int i = 0; i < size; i++) {
    digit[getDigit(arr[i], p)].push_back(arr[i]);
  }

  // for(int i=0;i<base;i++){
  //     std::cout<<"#"<<i<<"! ";
  //     printVec(digit[i]);
  // }
  int pos = 0;
  for (int d = base - 1; d >= 0; d--) {
    while (!digit[d].empty()) {
      arr[pos] = digit[d].back();
      digit[d].pop_back();
      pos++;
    }
  }
}

void radixSort(int* arr, int size) {
  for (int i = 0; i < digits; i++) {
    radix(arr, size, i);
  }
}

std::vector<int> getRandomVector(int size, int maxEl) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size == -1) {
    size = 1 + gen() % 20;
  }
  if (maxEl == -1) {
    maxEl = 1 + gen() % 40;
  }
  std::vector<int> res;
  for (int i = 0; i < size; i++) {
    res.push_back(gen() % maxEl - maxEl / 2);
  }
  return res;
}

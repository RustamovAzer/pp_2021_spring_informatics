// Copyright 2021 Makarychev Sergey
#include "../../../modules/task_1/makarychev_s_bitwise_sort/bitwise_sort.h"
#include <vector>
#include <random>

std::vector<int> getRandomVector(int sizeVec) {
    if (sizeVec < 0)
        throw "Wrong vector size";
    std::vector<int> rVec(sizeVec);
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_int_distribution<> uid(-1000, 2000);
    for (int i = 0; i < sizeVec; i++)
        rVec[i] = uid(mersenne);
    return rVec;
}

void createCounters(int* sortVec, int* counters, int sizeVec) {
  unsigned char* bytePointer = (unsigned char*)sortVec;
  unsigned char* dataEnd = (unsigned char*)(sortVec + sizeVec);
  int s = static_cast<int>(sizeof(int));
  while (bytePointer != dataEnd) {
    for (int i = 0; i < s; i++)
      counters[256 * i + *bytePointer++]++;
  }
}
void signedRadix(int byteNumber, int sizeVec, int* sourceVec, int* destVec, int* count) {
  int sum = 0;
  int* countPointer;
  if (byteNumber == sizeof(int) - 1) {
    int numNegative = 0;
    for (int i = 128; i < 256; i++)
      numNegative += count[i];

    sum = numNegative;
    countPointer = count;
    int tmp;
    for (int i = 0; i < 128; ++i, ++countPointer) {
      tmp = *countPointer;
      *countPointer = sum;
      sum += tmp;
    }
    sum = 0;
    countPointer = count + 128;
    for (int i = 0; i < 128; ++i, ++countPointer) {
      tmp = *countPointer;
      *countPointer = sum;
      sum += tmp;
    }
  } else {
    countPointer = count;
    int tmp;
    for (int i = 256; i > 0; --i, ++countPointer) {
      tmp = *countPointer;
      *countPointer = sum;
      sum += tmp;
    }
  }

  unsigned char* bytePointer = (unsigned char*)sourceVec + byteNumber;
  int* sourceVecPointer = sourceVec;
  for (int i = sizeVec; i > 0; --i, bytePointer += sizeof(int), ++sourceVecPointer) {
    countPointer = count + *bytePointer;
    destVec[*countPointer] = *sourceVecPointer;
    ++(*countPointer);
  }
}

void signedRadixSort(int* sortVec, int sizeVec) {
  int* out = new int[sizeVec];
  std::vector<int> counters(sizeof(int) * 256);
  int* count;
  createCounters(sortVec, counters.data(), sizeVec);
  int s = static_cast<int>(sizeof(int));
  for (int i = 0; i < s; i++) {
    count = counters.data() + 256 * i;
    signedRadix(i, sizeVec, sortVec, out, count);
    std::swap(sortVec, out);
  }
  delete[] out;
}

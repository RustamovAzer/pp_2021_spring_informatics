// Copyright 2021 Makarychev Sergey
#include <omp.h>
#include <random>
#include <algorithm>
#include <vector>
#include "../../../3rdparty/unapproved/unapproved.h"
#include "../../../modules/task_4/makarychev_s_bitwise_sort/bitwise_sort.h"

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

void mergeOrderVec(int* vec1, int size1,  int* vec2, int size2) {
    int* resVec = new int[size1 + size2];
    int i = 0, s = 0, j = 0;
    while (i < size1 && j < size2) {
        if (vec1[i] < vec2[j])
            resVec[s++] = vec1[i++];
        else
            resVec[s++] = vec2[j++];
    }
    while (i < size1)
        resVec[s++] = vec1[i++];
    while (j < size2)
        resVec[s++] = vec2[j++];
    i = s = 0;
    while (i < size1 + size2)
        vec1[i++] = resVec[s++];
    delete[] resVec;
}

void createCounters(std::vector<int>* sortVec, int* counters, int numByte) {
    unsigned char* bytePointer = (unsigned char*)sortVec->data();
    int s = static_cast<int>(sizeof(int));
    for (size_t i = 0; i < sortVec->size(); i++) {
        counters[bytePointer[s * i + numByte]]++;
    }
}

void countersSort(int numByte, std::vector<int>* sortVec,
    std::vector<int>* output) {
    unsigned char* bytePointer = (unsigned char*)sortVec->data();
    const int sizeCounter = 256;
    int counters[sizeCounter] = { 0 };
    int typeSize = static_cast<int>(sizeof(int));
    createCounters(sortVec, counters, numByte);
    int value = 0;
    for (int i = 0; i < sizeCounter; i++) {
        int tmp = counters[i];
        counters[i] = value;
        value += tmp;
    }
    int index = 0;
    for (size_t i = 0; i < sortVec->size(); i++) {
        index = bytePointer[typeSize * i + numByte];
        output->at(counters[index]++) = sortVec->at(i);
    }
}

void unsignedRadixSort(std::vector<int>* sortVec) {
    std::vector<int>* outbuf = new std::vector<int>(sortVec->size());
    for (int i = 0; i < 4; i++) {
        countersSort(i, sortVec, outbuf);
        std::swap(sortVec, outbuf);
    }
    delete outbuf;
}

void signedRadixSort(std::vector<int>* sortVec) {
    int positiveNum = 0;
    int negativeNum = 0;
    for (size_t i = 0; i < sortVec->size(); i++) {
        if (sortVec->at(i) < 0)
            negativeNum++;
        else
            positiveNum++;
    }
     std::vector<int> positiveNumVec(positiveNum);
    std::vector<int> negativeNumVec(negativeNum);

    int it1 = 0, it2 = 0;
    for (size_t i = 0; i < sortVec->size(); i++) {
        if (sortVec->at(i) >= 0)
            positiveNumVec[it1++] = sortVec->at(i);
        else
            negativeNumVec[it2++] = sortVec->at(i);
    }

    unsignedRadixSort(&positiveNumVec);
    unsignedRadixSort(&negativeNumVec);

    int current_buff = 0;
    for (size_t i = 0; i < negativeNumVec.size() ; i++) {
        sortVec->at(i) = negativeNumVec.at(current_buff);
        current_buff++;
    }
    current_buff = 0;
    for (size_t i = negativeNumVec.size();
        i < positiveNumVec.size() + negativeNumVec.size(); i++) {
        sortVec->at(i) = positiveNumVec.at(current_buff);
        current_buff++;
    }
}



void signedRadixSortParallel(std::vector<int>* sortVec,
    int leftIndex, int rightIndex, int sizeVec) {

    int positiveNum = 0;
    int negativeNum = 0;
    for (int i = leftIndex; i <= rightIndex; i++) {
        if (sortVec->at(i) < 0)
            negativeNum++;
        else
            positiveNum++;
    }
    std::vector<int> positiveNumVec(positiveNum);
    std::vector<int> negativeNumVec(negativeNum);

    int it1 = 0, it2 = 0;
    for (int i = leftIndex; i <= rightIndex; i++) {
        if (sortVec->at(i) >= 0)
            positiveNumVec[it1++] = sortVec->at(i);
        else
            negativeNumVec[it2++] = sortVec->at(i);
    }

    unsignedRadixSort(&positiveNumVec);
    unsignedRadixSort(&negativeNumVec);

    int currentId = 0;
    for (size_t i = leftIndex; i < negativeNumVec.size() + leftIndex; i++) {
        sortVec->at(i) = negativeNumVec.at(currentId);
        currentId++;
    }
    currentId = 0;
    for (size_t i = leftIndex + negativeNumVec.size();
        i < positiveNumVec.size() + leftIndex + negativeNumVec.size(); i++) {
        sortVec->at(i) = positiveNumVec.at(currentId);
        currentId++;
    }
}

void signedRadixSortStd(std::vector<int>* sortVec) {
    int numberThreads = 4;
    int sizeVec = sortVec->size();
    int sizePartVec = sizeVec / numberThreads;
    std::vector<std::thread> threads;
    int remainder = sizeVec % numberThreads;
    for (int i = 0; i < numberThreads; i++) {
        std::thread th([&sortVec, &sizePartVec,
            &remainder](int i) {
            if (i == 0) {
                signedRadixSortParallel(sortVec, i,
                    (i + 1) * sizePartVec + remainder - 1, sizePartVec);
            } else {
                signedRadixSortParallel(sortVec, i * sizePartVec + remainder,
                    (i + 1) * sizePartVec + remainder - 1, sizePartVec);
            }
            }, i);
        threads.emplace_back(std::move(th));
    }

    for (int i = 0; i < numberThreads; i++)
        threads[i].join();
    for (int i = 1; i < numberThreads; i++) {
        int remainder = sizeVec % numberThreads;
        int current_size = sortVec->size() / numberThreads;
        if (i == 0) {
            current_size = sortVec->size() /
                numberThreads + sortVec->size() % numberThreads;
        } else {
            current_size = sortVec->size() / numberThreads;
        }
        mergeOrderVec(sortVec->data(), current_size * i + remainder,
            sortVec->data() + current_size * i + remainder, current_size);
    }
}

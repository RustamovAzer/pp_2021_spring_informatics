// Copyright 2021 Belik Julia
#ifndef MODULES_TASK_3_BELIK_J_RADIX_SORT_RADIXSORTB_H_
#define MODULES_TASK_3_BELIK_J_RADIX_SORT_RADIXSORTB_H_

#include <vector>
#include <iostream>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

void RadixSort(double* vec, size_t len, double* vec2);
std::vector<double> Vector(size_t n, double a, double b);
std::vector<double> MergeBatcherPar(std::vector<double> vec, size_t size, int thr);
std::vector<double> MergeBatcherSeq(std::vector<double> vec, size_t size, int thr);
void Shuffle(double* vec, size_t len, double* vec2);
void PMerge(double* start1, double* start2, double* start3, size_t len1, size_t len2);

class tbb_radixsort {
 private:
    double *in;
    double *out;
    std::vector<size_t> offsets;
    std::vector<size_t> lens;
 public:
     tbb_radixsort(double *in_, double *out_, const std::vector<size_t>& offsets_,
         const std::vector<size_t>& lens_) : in(in_), out(out_), offsets(offsets_), lens(lens_) {}
    void operator() (const tbb::blocked_range<int> &range) const {
        RadixSort(in + offsets[range.begin()], lens[range.begin()], out + offsets[range.begin()]);
    }
};

class tbb_pmerge {
 private:
    double *in;
    double *out;
    std::vector<size_t> offsets;
    std::vector<size_t> lens;
    int mergecount;
    int offset;
    int thr;

 public:
     tbb_pmerge(double *in_, double *out_, const std::vector<size_t>& offsets_, const std::vector<size_t>& lens_,
         int mergecount_, int offset_, int thr_) : in(in_), out(out_), offsets(offsets_), lens(lens_),
         mergecount(mergecount_), offset(offset_), thr(thr_) {}
    void operator() (const tbb::blocked_range<int> &range) const {
        if ((range.begin() % mergecount == 0) && (range.begin() + offset < thr)) {
            size_t len1 = lens[range.begin()] / 2 + lens[range.begin()] % 2;
            size_t len2 = lens[range.begin() + offset] / 2 + lens[range.begin() + offset] % 2;
            PMerge(in + offsets[range.begin()], in + offsets[range.begin() + offset],
                out + offsets[range.begin()], len1, len2);
        }
        if ((range.begin() - offset >= 0) && ((range.begin() - offset) % mergecount == 0)) {
            size_t start3 = offsets[range.begin() - offset] + lens[range.begin() - offset] / 2
                + lens[range.begin() - offset] % 2 + lens[range.begin()] / 2 + lens[range.begin()] % 2;
            size_t start1 = offsets[range.begin() - offset] + lens[range.begin() - offset] / 2
                + lens[range.begin() - offset] % 2;
            size_t start2 = offsets[range.begin()] + lens[range.begin()] / 2 + lens[range.begin()] % 2;
            PMerge(in + start1, in + start2, out + start3, lens[range.begin() - offset] / 2, lens[range.begin()] / 2);
        }
    }
};

class tbb_ppmerge {
 private:
    double *in;
    double *out;
    std::vector<size_t> offsets;
    std::vector<size_t> lens;
    int mergecount;
    int offset;
    int thr;
 public:
     tbb_ppmerge(double *in_, double *out_, const std::vector<size_t>& offsets_, const std::vector<size_t>& lens_,
         int mergecount_, int offset_, int thr_) : in(in_), out(out_), offsets(offsets_), lens(lens_),
         mergecount(mergecount_), offset(offset_), thr(thr_) {}
    void operator() (const tbb::blocked_range<int> &range) const {
        if ((range.begin() % mergecount == 0) && (range.begin() + offset < thr)) {
            size_t evencount = lens[range.begin()] / 2 + lens[range.begin() + offset] / 2 + lens[range.begin()] % 2
                + lens[range.begin() + offset] % 2;
            size_t oddcount = lens[range.begin()] / 2 + lens[range.begin() + offset] / 2;
            PMerge(in + offsets[range.begin()], in + offsets[range.begin()] + evencount, out + offsets[range.begin()],
                evencount, oddcount);
        }
    }
};

class tbb_shuffle {
 private:
    double *in;
    double *out;
    std::vector<size_t> offsets;
    std::vector<size_t> lens;
 public:
    tbb_shuffle(double *in_, double *out_, const std::vector<size_t>& offsets_, const std::vector<size_t>& lens_)
        : in(in_), out(out_), offsets(offsets_), lens(lens_) {}
    void operator() (const tbb::blocked_range<int> &range) const {
        if (lens[range.begin()] != 0) {
            Shuffle(in + offsets[range.begin()], lens[range.begin()], out + offsets[range.begin()]);
        }
    }
};

class tbb_sshuffle {
 private:
    double *in;
    double *out;
    std::vector<size_t> offsets;
    std::vector<size_t> lens;
    int mergecount;
    int offset;
    int thr;
 public:
    tbb_sshuffle(double *in_, double *out_, const std::vector<size_t>& offsets_, const std::vector<size_t>& lens_,
        int mergecount_, int offset_, int thr_) : in(in_), out(out_), offsets(offsets_), lens(lens_),
        mergecount(mergecount_), offset(offset_), thr(thr_) {}
    void operator() (const tbb::blocked_range<int> &range) const {
        if ((range.begin() % mergecount == 0) && (range.begin() + offset < thr)) {
            Shuffle(in + offsets[range.begin()], lens[range.begin()], out + offsets[range.begin()]);
        }
    }
};

#endif  // MODULES_TASK_3_BELIK_J_RADIX_SORT_RADIXSORTB_H_

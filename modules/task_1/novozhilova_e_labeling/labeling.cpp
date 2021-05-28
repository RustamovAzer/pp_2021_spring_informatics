// Copyright 2021 Novozhilova Ekaterina
#include <algorithm>
#include "../../../modules/task_1/novozhilova_e_labeling/labeling.h"

std::vector<std::vector<int>> Labeling(std::vector<std::vector<int>> arr, int rows, int cols) {
    int label = 1;
     std::vector<std::vector<int>> lab;
     lab.resize(rows);
     for (int i = 0; i < rows; i++) {
          lab[i].resize(cols);
        for (int j = 0; j < cols; j++) {
               lab[i][j] = 0;
          }
     }
     if (arr[0][0] == 1) {
          lab[0][0] = label;
     } else {
          label = 0;
     }
     int neigh;
     for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
               if (arr[i][j] == 1) {
                    neigh = 2;
                    if (i != 0) {
                         if (arr[i - 1][j] == 1) {
                              neigh++;
                              lab[i][j] = lab[i - 1][j];
                         } else {
                              neigh -= 1;
                         }
                    }
                    if (j != 0) {
                         if (arr[i][j - 1] == 1) {
                              neigh++;
                              lab[i][j] = lab[i][j - 1];
                         } else {
                              neigh -= 1;
                         }
                    }
                    if (neigh < 2) {
                         label++;
                         lab[i][j] = label;
                    }
                    if (neigh == 4) {
                         int minim = std::min(lab[i][j - 1], lab[i - 1][j]);
                         lab[i][j] = minim;
                         lab[i][j - 1] = minim;
                         lab[i - 1][j] = minim;
                         label = minim;
                    }
               }
          }
     }
     for (int i = rows - 2; i > 0; i--) {
          for (int j = cols - 2; j > 0; j--) {
               if (lab[i][j] != lab[i][j - 1]) {
                    int mim = std::min(lab[i][j], lab[i][j - 1]);
                    if (mim != 0) {
                         lab[i][j] = lab[i][j - 1] = mim;
                    }
               }
               if (lab[i][j] != lab[i - 1][j]) {
                    int mim = std::min(lab[i][j], lab[i - 1][j]);
                    if (mim != 0) {
                         lab[i][j] = lab[i - 1][j] = mim;
                    }
               }
          }
     }
     int max_lab = 1;
     for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
               if (lab[i][j] > max_lab) {
                    max_lab = lab[i][j];
               }
          }
     }
    return lab;
}

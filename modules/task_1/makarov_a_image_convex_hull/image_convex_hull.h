// Copyright 2021 Makarov Alexander
#ifndef MODULES_TASK_1_MAKAROV_A_IMAGE_CONVEX_HULL_IMAGE_CONVEX_HULL_H_
#define MODULES_TASK_1_MAKAROV_A_IMAGE_CONVEX_HULL_IMAGE_CONVEX_HULL_H_

#include <vector>
#include <list>
#include <utility>

std::vector<int> mark_components(const std::vector<int>& bin_image, int w,
                                 int h);
std::vector<std::list <std::pair<int, int> > > get_convex_hulls(
                           const std::vector<int>& marked_image, int w, int h);

#endif  // MODULES_TASK_1_MAKAROV_A_IMAGE_CONVEX_HULL_IMAGE_CONVEX_HULL_H_

// Copyright 2021 Makarov Alexander
#include "../../../modules/task_1/makarov_a_image_convex_hull/image_convex_hull.h"

#include <utility>
#include <stack>
#include <iostream>
#include <list>
#include <algorithm>

std::vector<int> mark_components(const std::vector<int>& bin_image,
                                 int w, int h) {
    int components_count = 1;  // plus background
    std::vector<int> result(bin_image);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (result[i * w + j] == 0) {
                components_count++;
                std::pair<int, int> burn_point(j, i);
                result[i * w + j] = components_count;
                std::stack<std::pair<int, int> > burn_stack;
                burn_stack.push(burn_point);
                while (!burn_stack.empty()) {
                    std::pair<int, int> curr_point = burn_stack.top();
                    int x = curr_point.first;
                    int y = curr_point.second;
                    burn_stack.pop();
                    if (y > 0 &&
                        result[(y - 1) * w + x] == 0) {
                        result[(y - 1) * w + x] = components_count;
                        burn_stack.push(std::pair<int, int>(x, y - 1));
                    }
                    if (y + 1 < h &&
                        result[(y + 1) * w + x] == 0) {
                        result[(y + 1) * w + x] = components_count;
                        burn_stack.push(std::pair<int, int>(x, y + 1));
                    }
                    if (x > 0 &&
                        result[y * w + x - 1] == 0) {
                        result[y * w + x - 1] = components_count;
                        burn_stack.push(std::pair<int, int>(x - 1, y));
                    }
                    if (x + 1 < w &&
                        result[y * w + x + 1] == 0) {
                        result[y * w + x + 1] = components_count;
                        burn_stack.push(std::pair<int, int>(x + 1, y));
                    }
                }
            }
        }
    }
    return result;
}

int orientation(std::pair<int, int> p, std::pair<int, int> q,
                std::pair<int, int> r) {
    int val = (q.second - p.second) * (r.first - q.first) -
              (q.first - p.first) * (r.second - q.second);
    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2;  // clock or counterclock wise
}

std::vector<std::list <std::pair<int, int> > > get_convex_hulls(
                          const std::vector<int>& marked_image, int w, int h) {
    int comp_count = *std::max_element(marked_image.begin(),
                                       marked_image.end()) - 1;
    std::vector<std::list<std::pair<int, int> > > components(comp_count);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (marked_image[i * w + j] == 1) {
                continue;
            } else {
                std::pair<int, int> point(j, i);
                components[marked_image[i * w + j] - 2].push_back(point);
            }
        }
    }
    std::vector<std::list<std::pair<int, int> > > result(comp_count);
    int counter = 0;
    for (std::list<std::pair<int, int> > component_list : components) {
        if (component_list.size() < 3) {
            result[counter] = component_list;
        } else {
            std::pair<int, int> start(w, h);
            int start_idx = 0;
            int n = component_list.size();
            std::vector<std::pair<int, int> > component(n);
            int counter_1 = 0;
            for (auto point : component_list) {
                component[counter_1] = point;
                if (point.first < start.first) {
                    start = point;
                    start_idx = counter_1;
                }
                counter_1++;
            }
            int curr = start_idx;
            int next;
            do {
                result[counter].push_back(component[curr]);
                next = (curr + 1) % n;
                // next = 0;
                for (int i = 0; i < n; i++) {
                    if (orientation(component[curr], component[i],
                                    component[next]) == 2)
                        next = i;
                }
                curr = next;
            } while (curr != start_idx);
        }
        counter++;
    }
    return result;
}




// Laboratory work No. 4. Variant 2: Parallel QuickSort (OpenMP).
// Compile: g++ -std=c++17 -O3 -fopenmp -Wall -Wextra main.cpp -o lab4
// Run:     ./lab4 --quick   (or ./lab4 for the full experiment)

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr int INSERTION_LIMIT = 24;
constexpr int TASK_GRAIN = 16384;

int median3(int a, int b, int c) {
    if (a < b) return (b < c) ? b : std::max(a, c);
    return (a < c) ? a : std::max(b, c);
}

void insertionSort(std::vector<int>& a, int left, int right) {
    for (int i = left + 1; i <= right; ++i) {
        int value = a[i];
        int j = i - 1;
        while (j >= left && a[j] > value) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = value;
    }
}

// Hoare-style partition: a[left..j] <= pivot <= a[i..right].
std::pair<int, int> partitionRange(std::vector<int>& a, int left, int right) {
    const int mid = left + (right - left) / 2;
    const int pivot = median3(a[left], a[mid], a[right]);
    int i = left;
    int j = right;
    while (i <= j) {
        while (a[i] < pivot) ++i;
        while (a[j] > pivot) --j;
        if (i <= j) {
            std::swap(a[i], a[j]);
            ++i;
            --j;
        }
    }
    return {j, i};
}

void quickSortSerial(std::vector<int>& a, int left, int right) {
    while (left < right) {
        if (right - left + 1 <= INSERTION_LIMIT) {
            insertionSort(a, left, right);
            return;
        }
        auto [j, i] = partitionRange(a, left, right);
        // Sort the smaller side recursively to keep the call stack shallow.
        if (j - left < right - i) {
            if (left < j) quickSortSerial(a, left, j);
            left = i;
        } else {
            if (i < right) quickSortSerial(a, i, right);
            right = j;
        }
    }
}

void quickSortTasks(std::vector<int>& a, int left, int right, int depthLeft) {
    if (left >= right) return;
    if (depthLeft <= 0 || right - left + 1 < TASK_GRAIN) {
        quickSortSerial(a, left, right);
        return;
    }

    auto [j, i] = partitionRange(a, left, right);
    const int nextDepth = depthLeft - 1;

    // Each task receives a disjoint segment of the same array. No data race.
    if (left < j) {
        #pragma omp task shared(a) firstprivate(left, j, nextDepth)
        quickSortTasks(a, left, j, nextDepth);
    }
    if (i < right) quickSortTasks(a, i, right, nextDepth);

    // Do not finish this recursion level until its child task completes.
    #pragma omp taskwait
}

void quickSortParallel(std::vector<int>& a, int threads) {
    if (a.size() < 2) return;
    int depth = 0;
    for (int budget = 1; budget < threads * 4; budget *= 2) ++depth;

    // Exactly one initial task creator; workers pick tasks from OpenMP runtime.
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        quickSortTasks(a, 0, static_cast<int>(a.size()) - 1, depth);
    }
}

std::vector<int> makeInput(int n) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(2026u + n));
    std::uniform_int_distribution<int> dist(-1000000, 1000000);
    std::vector<int> data(static_cast<size_t>(n));
    for (int& v : data) v = dist(rng);
    return data;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t k = values.size() / 2;
    return (values.size() % 2) ? values[k] : (values[k - 1] + values[k]) / 2.0;
}

struct Measurement {
    int n;
    int threads;
    double serialMs;
    double parallelMs;
    double speedup;
};

// Minimal self-contained SVG graphs: no Python packages needed.
void writeChart(const std::vector<Measurement>& values,
                const std::vector<int>& sizes,
                const std::vector<int>& threadCounts,
                bool showSpeedup,
                const std::string& filename) {
    const int W = 1100, H = 660;
    const double x0 = 105, y0 = 95, plotW = 720, plotH = 455;
    const char* palette[] = {"#2563eb", "#16a34a", "#e11d48", "#9333ea", "#ea580c", "#0891b2"};
    const std::string title = showSpeedup
        ? "Parallel QuickSort | Speedup vs array size"
        : "Parallel QuickSort | Execution time vs array size";

    double yMax = showSpeedup ? 2.0 : 1.0;
    for (const auto& v : values) {
        yMax = std::max(yMax, showSpeedup ? v.speedup : v.parallelMs);
        if (!showSpeedup) yMax = std::max(yMax, v.serialMs);
    }
    yMax *= 1.13;

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open chart file: " + filename);
    out << std::fixed << std::setprecision(2);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << W << ' ' << H << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    out << "<text x=\"" << x0 << "\" y=\"45\" font-family=\"Arial,sans-serif\" font-size=\"25\" font-weight=\"bold\" fill=\"#111827\">" << title << "</text>\n";

    auto px = [&](size_t index) { return x0 + (sizes.size() == 1 ? plotW / 2 : plotW * index / (sizes.size() - 1)); };
    auto py = [&](double value) { return y0 + plotH * (1.0 - value / yMax); };

    for (int tick = 0; tick <= 5; ++tick) {
        double value = yMax * tick / 5.0;
        double y = py(value);
        out << "<line x1=\"" << x0 << "\" y1=\"" << y << "\" x2=\"" << x0 + plotW << "\" y2=\"" << y << "\" stroke=\"#e5e7eb\"/>\n";
        out << "<text x=\"" << x0 - 16 << "\" y=\"" << y + 5 << "\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"13\" fill=\"#475569\">" << value << "</text>\n";
    }
    for (size_t k = 0; k < sizes.size(); ++k) {
        const int n = sizes[k];
        const std::string label = n >= 1000000 ? std::to_string(n / 1000000) + "M"
             : n >= 1000 ? std::to_string(n / 1000) + "k" : std::to_string(n);
        out << "<text x=\"" << px(k) << "\" y=\"" << y0 + plotH + 29 << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"13\" fill=\"#475569\">" << label << "</text>\n";
    }
    out << "<text x=\"" << x0 + plotW / 2 << "\" y=\"" << H - 43 << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\" fill=\"#334155\">Array size (n)</text>\n";
    out << "<text x=\"27\" y=\"" << y0 + plotH / 2 << "\" transform=\"rotate(-90 27 " << y0 + plotH / 2 << ")\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\" fill=\"#334155\">" << (showSpeedup ? "Speedup S = T1 / Tm" : "Time (milliseconds)") << "</text>\n";

    for (size_t t = 0; t < threadCounts.size(); ++t) {
        const int threads = threadCounts[t];
        const std::string color = palette[t % 6];
        out << "<polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"3\" points=\"";
        for (size_t k = 0; k < sizes.size(); ++k) {
            for (const auto& m : values) if (m.n == sizes[k] && m.threads == threads) {
                out << px(k) << ',' << py(showSpeedup ? m.speedup : m.parallelMs) << ' ';
                break;
            }
        }
        out << "\"/>\n";
        for (size_t k = 0; k < sizes.size(); ++k) {
            for (const auto& m : values) if (m.n == sizes[k] && m.threads == threads) {
                out << "<circle cx=\"" << px(k) << "\" cy=\"" << py(showSpeedup ? m.speedup : m.parallelMs) << "\" r=\"4.5\" fill=\"" << color << "\"/>\n";
                break;
            }
        }
        const double lx = x0 + plotW + 46, ly = 150 + 37 * t;
        out << "<line x1=\"" << lx << "\" y1=\"" << ly << "\" x2=\"" << lx + 28 << "\" y2=\"" << ly << "\" stroke=\"" << color << "\" stroke-width=\"4\"/>\n";
        out << "<text x=\"" << lx + 40 << "\" y=\"" << ly + 5 << "\" font-family=\"Arial\" font-size=\"15\" fill=\"#1f2937\">" << threads << " threads</text>\n";
    }
    if (!showSpeedup) {
        out << "<polyline fill=\"none\" stroke=\"#111827\" stroke-dasharray=\"8,6\" stroke-width=\"2.5\" points=\"";
        for (size_t k = 0; k < sizes.size(); ++k) {
            for (const auto& m : values) if (m.n == sizes[k]) {
                out << px(k) << ',' << py(m.serialMs) << ' ';
                break;
            }
        }
        out << "\"/>\n";
        out << "<line x1=\"" << x0 + plotW + 46 << "\" y1=\"" << 150 + 37 * threadCounts.size() << "\" x2=\"" << x0 + plotW + 74 << "\" y2=\"" << 150 + 37 * threadCounts.size() << "\" stroke=\"#111827\" stroke-width=\"3\" stroke-dasharray=\"8,6\"/>\n";
        out << "<text x=\"" << x0 + plotW + 86 << "\" y=\"" << 155 + 37 * threadCounts.size() << "\" font-family=\"Arial\" font-size=\"15\" fill=\"#1f2937\">Serial</text>\n";
    }
    out << "<text x=\"" << x0 << "\" y=\"" << H - 14 << "\" font-family=\"Arial\" font-size=\"12\" fill=\"#64748b\">Measurements depend on CPU cores, runtime load and compiler settings.</text>\n";
    out << "</svg>\n";
}

void runSelfTests() {
    const std::vector<std::vector<int>> testCases = {
        {}, {9}, {2, 1}, {1, 2}, {5, 5, 5, 5},
        {4, -3, 4, 0, 99, -3, 2, 2, 1},
    };
    for (const auto& input : testCases) {
        auto expected = input;
        std::sort(expected.begin(), expected.end());
        auto serial = input;
        if (!serial.empty()) quickSortSerial(serial, 0, static_cast<int>(serial.size()) - 1);
        if (serial != expected) throw std::runtime_error("Serial self-test failed");
        for (int threads : {2, 4, 6, 8, 10, 20}) {
            auto parallel = input;
            quickSortParallel(parallel, threads);
            if (parallel != expected) throw std::runtime_error("Parallel self-test failed");
        }
    }
    // Test task creation and concurrent writes with a large, duplicate-heavy array.
    auto input = makeInput(100000);
    for (size_t i = 0; i < input.size(); i += 3) input[i] = 42;
    auto expected = input;
    std::sort(expected.begin(), expected.end());
    for (int threads : {2, 4, 6, 8, 10, 20}) {
        auto parallel = input;
        quickSortParallel(parallel, threads);
        if (parallel != expected) throw std::runtime_error("Large self-test failed");
    }
    std::cout << "Self-tests: PASSED (including repeated values and large arrays).\n";
}

int main(int argc, char* argv[]) {
    try {
        std::vector<int> sizes = {10000, 50000, 100000, 250000, 500000, 1000000, 2000000};
        const std::vector<int> threadCounts = {2, 4, 6, 8, 10, 20};
        int repeats = 3;
        bool onlyTest = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--quick") {
                sizes = {10000, 50000, 200000};
                repeats = 1;
            } else if (arg == "--test") {
                onlyTest = true;
            } else if (arg == "--help") {
                std::cout << "Usage: ./lab4 [--quick | --test | --help]\n"
                             "  no args: 7 array sizes, 6 thread counts, 3 repetitions\n"
                             "  --quick: 3 array sizes, 6 thread counts, 1 repetition\n"
                             "  --test: only check correctness\n";
                return 0;
            } else {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        omp_set_dynamic(0); // OpenMP should not silently shrink requested teams.
        std::cout << "Laboratory #4 | Variant 2 | OpenMP Parallel QuickSort\n";
        std::cout << "Available processors reported by OpenMP: " << omp_get_num_procs() << "\n";
        runSelfTests();
        if (onlyTest) return 0;
        fs::create_directories("results");

        std::vector<Measurement> measurements;
        for (int n : sizes) {
            std::vector<int> source = makeInput(n);
            auto expected = source;
            std::sort(expected.begin(), expected.end());

            std::vector<double> serialTimes;
            for (int trial = 0; trial < repeats; ++trial) {
                auto copy = source;   // Copying is outside the timed section.
                const double start = omp_get_wtime();
                quickSortSerial(copy, 0, n - 1);
                const double elapsed = (omp_get_wtime() - start) * 1000.0;
                if (copy != expected) throw std::runtime_error("Serial sort output incorrect");
                serialTimes.push_back(elapsed);
            }
            const double serialMs = median(serialTimes);
            std::cout << "\nn = " << n << " | serial = " << std::fixed << std::setprecision(3) << serialMs << " ms\n";

            for (int threads : threadCounts) {
                std::vector<double> parallelTimes;
                for (int trial = 0; trial < repeats; ++trial) {
                    auto copy = source;
                    const double start = omp_get_wtime();
                    quickSortParallel(copy, threads);
                    const double elapsed = (omp_get_wtime() - start) * 1000.0;
                    if (copy != expected) {
                        throw std::runtime_error("Parallel sort output incorrect for " + std::to_string(threads) + " threads");
                    }
                    parallelTimes.push_back(elapsed);
                }
                const double parallelMs = median(parallelTimes);
                const double speedup = serialMs / parallelMs;
                measurements.push_back({n, threads, serialMs, parallelMs, speedup});
                std::cout << "  m = " << std::setw(2) << threads
                          << " | time = " << std::setw(10) << parallelMs << " ms"
                          << " | S = " << std::setprecision(3) << speedup << "x\n";
            }
        }

        std::ofstream csv("results/benchmark.csv");
        if (!csv) throw std::runtime_error("Cannot write results/benchmark.csv");
        csv << "n,threads,serial_ms,parallel_ms,speedup,efficiency\n";
        csv << std::fixed << std::setprecision(6);
        for (const auto& m : measurements) {
            csv << m.n << ',' << m.threads << ',' << m.serialMs << ',' << m.parallelMs
                << ',' << m.speedup << ',' << m.speedup / m.threads << '\n';
        }
        csv.close();

        writeChart(measurements, sizes, threadCounts, true, "results/speedup.svg");
        writeChart(measurements, sizes, threadCounts, false, "results/execution_times.svg");
        std::cout << "\nAll results validated. Created files:\n"
                     "  results/benchmark.csv\n"
                     "  results/speedup.svg\n"
                     "  results/execution_times.svg\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

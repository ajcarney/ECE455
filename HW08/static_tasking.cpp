#include <taskflow/taskflow.hpp> // Taskflow header
#include <cstdio>                // For printf

int main() {

    // Create an executor
    tf::Executor executor;

    // Create a taskflow
    tf::Taskflow taskflow("Static Taskflow Demo");

    // Create tasks
    auto A = taskflow.emplace([]() { printf("Task A\n"); });
    auto B = taskflow.emplace([]() { printf("Task B\n"); });
    auto C = taskflow.emplace([]() { printf("Task C\n"); });
    auto D = taskflow.emplace([]() { printf("Task D\n"); });

    // Define dependencies: A -> B, A -> C, B -> D, C -> D
    A.precede(B, C);
    B.precede(D);
    C.precede(D);

    // Run the taskflow
    executor.run(taskflow).wait();

    return 0;
}


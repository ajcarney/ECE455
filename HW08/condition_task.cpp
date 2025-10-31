#include <taskflow/taskflow.hpp>
#include <cstdio>

int main() {

    tf::Executor executor;
    tf::Taskflow taskflow("Condition Task Demo");

    int counter = 0;
    const int limit = 5;

    // Initialize counter
    auto init = taskflow.emplace([&]() {
        printf("Initialize counter = %d\n", counter);
    });

    // Loop task with condition
    auto loop = taskflow.emplace([&]() -> int {
        printf("Loop iteration %d\n", counter);
        counter++;
        return (counter < limit) ? 0 : 1;  // 0 => go back, 1 => exit
    }).name("cond");

    // Done task
    auto done = taskflow.emplace([]() {
        printf("Loop done.\n");
    });

    // Define dependencies
    init.precede(loop);
    loop.precede(loop, done);  // self-edge enables iteration

    // Run taskflow
    executor.run(taskflow).wait();

    return 0;
}


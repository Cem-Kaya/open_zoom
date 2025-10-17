#include <cstdlib>
#include <iostream>
#include <exception>

#ifdef _WIN32
#include "openzoom/app/app.hpp"
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    try {
        openzoom::OpenZoomApp app(argc, argv);
        return app.Run();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
#else
    std::cerr << "OpenZoom currently supports Windows with CUDA, Qt, and Direct3D12 only.\n";
    return EXIT_FAILURE;
#endif
}

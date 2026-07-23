find_package(Qt6 REQUIRED COMPONENTS Core Gui Test)

add_executable(settings_store_tests
    settings_store_tests.cpp
    ${PROJECT_SOURCE_DIR}/src/app/color_schemes.cpp
    ${PROJECT_SOURCE_DIR}/src/app/settings_store.cpp
)

target_include_directories(settings_store_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(settings_store_tests PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
)

add_test(NAME settings_store_round_trip COMMAND settings_store_tests)
get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
get_filename_component(QT_BIN_DIRECTORY "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
set_tests_properties(settings_store_round_trip PROPERTIES
    TIMEOUT 30
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${QT_BIN_DIRECTORY}"
)

add_executable(view_transform_tests
    view_transform_tests.cpp
    ${PROJECT_SOURCE_DIR}/src/common/view_transform.cpp
)

target_include_directories(view_transform_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(view_transform_tests PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
)

add_test(NAME viewport_transform_geometry COMMAND view_transform_tests)
set_tests_properties(viewport_transform_geometry PROPERTIES
    TIMEOUT 30
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${QT_BIN_DIRECTORY}"
)

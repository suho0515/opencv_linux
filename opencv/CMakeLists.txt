cmake_minimum_required(VERSION 2.8.3)
project(opencv)

find_package(catkin REQUIRED COMPONENTS
 roscpp
 genmsg
 geometry_msgs
 message_generation
 std_msgs
 cv_bridge
 sensor_msgs
 image_transport
)

find_package(OpenCV REQUIRED)
# target_link_libraries(opencv ${OpenCV_LIBRARIES})

## messages
add_message_files(FILES MsgCenter.msg)
add_message_files(FILES MsgAngle.msg)
add_message_files(FILES MsgDetection.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
LIBRARIES opencv
CATKIN_DEPENDS std_msgs roscpp message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(image_converter src/image_converter.cpp)
add_executable(squares src/squares.cpp)

add_dependencies(image_converter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(squares opencv_generate_messages_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(image_converter ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(squares ${catkin_LIBRARIES} ${OpenCV_LIBS})

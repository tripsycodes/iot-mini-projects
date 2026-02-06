#include "ModbusMaster.h"
#include "util/stream.h"
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

const char *device = "/dev/ttyUSB0";
const uint16_t baud = 4800;
const char *named_pipe = "soil.sock";

Stream stream(device, baud);

typedef struct response {
    time_t time;
    float moisture;
    float temperature;
    int ec;
    float ph;
    int nitrogen;
    int phosphorus;
    int potassium;
    float salinity;
    int tds;
} Response_t;

void preTransmission() {
    // NOTE: called before any frame transmission
}
void postTransmission() {
    // NOTE: called after any frame transmission
}

void printResponse(const Response_t &r);

auto main(int argc, char *argv[]) -> int {

    signal(SIGPIPE, SIG_IGN);

    std::cout << "Waiting to open pipe" << std::endl;
    mkfifo(named_pipe, 0666); // make a named pipe for reading and writing
    int pipe_fd = -1;

    ModbusMaster node;
    node.begin(1, stream);
    node.preTransmission(preTransmission);
    node.postTransmission(postTransmission);

    uint8_t result;
    while (true) {
        if (pipe_fd == -1) {
            pipe_fd = open(named_pipe, O_WRONLY | O_NONBLOCK);
            if (pipe_fd == -1) {
                perror("open pipe");
            }
        }
        result = node.readHoldingRegisters(0x0000, 9);
        if (result == node.ku8MBSuccess) {
            Response_t response;
            // obtain the seconds since unix epoch
            const auto p1 = std::chrono::system_clock::now();
            response.time = std::chrono::duration_cast<std::chrono::seconds>(
                                p1.time_since_epoch())
                                .count();

            response.moisture = node.getResponseBuffer(0) / 10.0;
            response.temperature = node.getResponseBuffer(1) / 10.0;
            response.ec = node.getResponseBuffer(2);
            response.ph = node.getResponseBuffer(3) / 10.0;
            response.nitrogen = node.getResponseBuffer(4);
            response.phosphorus = node.getResponseBuffer(5);
            response.potassium = node.getResponseBuffer(6);
            response.salinity = node.getResponseBuffer(7) / 10.0;
            response.tds = node.getResponseBuffer(8);

            ssize_t n =
                dprintf(pipe_fd, "%ld,%.1f,%.1f,%d,%.1f,%d,%d,%d,%.1f,%d\n",
                        response.time, response.moisture, response.temperature,
                        response.ec, response.ph, response.nitrogen,
                        response.phosphorus, response.potassium,
                        response.salinity, response.tds);

            if (n == -1) {
                perror("write");
                close(pipe_fd);
                pipe_fd = -1;
            }
        } else {
            std::cout << "Failure, code = " << static_cast<int>(result)
                      << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    close(pipe_fd);
    return 0;
}

void printResponse(const Response_t &r) {
    char timebuf[32];
    std::tm *tm_info = std::localtime(&r.time);
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", tm_info);

    std::cout << "================ Soil Sensor Reading ================\n";
    std::cout << "Time        : " << timebuf << '\n';
    std::cout << "Moisture    : " << r.moisture << " %\n";
    std::cout << "Temperature : " << r.temperature << " °C\n";
    std::cout << "EC          : " << r.ec << " µS/cm\n";
    std::cout << "pH          : " << r.ph << '\n';
    std::cout << "Nitrogen    : " << r.nitrogen << " mg/kg\n";
    std::cout << "Phosphorus  : " << r.phosphorus << " mg/kg\n";
    std::cout << "Potassium   : " << r.potassium << " mg/kg\n";
    std::cout << "Salinity    : " << r.salinity << " ppt\n";
    std::cout << "TDS         : " << r.tds << " ppm\n";
    std::cout << "=====================================================\n";
}

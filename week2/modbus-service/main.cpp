#include "ModbusMaster.h"
#include "util/stream.h"
#include <iostream>
#include <string.h>

Stream stream("/dev/ttyUSB0", 4800);

void preTransmission() { std::cout << "Before Transmission" << std::endl; }

void postTransmission() { std::cout << "After Transmission" << std::endl; }

auto main(int argc, char *argv[]) -> int {

    ModbusMaster node;
    node.begin(1, stream);
    node.preTransmission(preTransmission);
    node.postTransmission(postTransmission);

    uint8_t result;
    result = node.readHoldingRegisters(0x0000, 9);
    if (result == node.ku8MBSuccess) {
        float moisture = node.getResponseBuffer(0) / 10.0;
        float temperature = node.getResponseBuffer(1) / 10.0;
        int ec = node.getResponseBuffer(2);
        float ph = node.getResponseBuffer(3) / 10.0;

        int nitrogen = node.getResponseBuffer(4);
        int phosphorus = node.getResponseBuffer(5);
        int potassium = node.getResponseBuffer(6);

        float salinity = node.getResponseBuffer(7) / 10.0;
        int tds = node.getResponseBuffer(8);

        std::cout << moisture << ' ' << temperature << std::endl;
    } else {
        std::cout << "Failure" << std::endl;
    }

    return 0;
}

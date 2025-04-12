#include <array>
#include <vector>
#include <iostream>
#include <tbb/flow_graph.h>
#include <tbb/global_control.h>
#include <sycl/sycl.hpp>
#include <pcap.h>
#include <cstring>
#include <netinet/ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>
#include "dpc_common.hpp"

constexpr size_t burst_size = 32;
constexpr size_t max_packet_size = 1518;
constexpr size_t packet_fields = max_packet_size;

using PacketMatrix = std::array<std::array<uint8_t, packet_fields>, burst_size>;

bool is_ipv4(const uint8_t* packet) {
    return (packet[12] == 0x08 && packet[13] == 0x00);
}

bool is_ipv6(const uint8_t* packet) {
    return (packet[12] == 0x86 && packet[13] == 0xDD);
}

void increment_ipv4_dst_ip(uint8_t* packet) {
    for (int i = 0; i < 4; ++i) {
        packet[30 + i] += 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <interface_name>\n";
        return 1;
    }

    const char* interface_name = argv[1];

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* pcap = pcap_open_live(interface_name, BUFSIZ, 1, 1000, errbuf);
    if (!pcap) {
        std::cerr << "Error opening interface " << interface_name << ": " << errbuf << std::endl;
        return 1;
    }

    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 4);
    tbb::flow::graph g;

    sycl::queue q(sycl::default_selector_v, dpc_common::exception_handler,
                  sycl::property_list{sycl::property::queue::enable_profiling()});

    tbb::flow::input_node<PacketMatrix> input_node{g, [&](tbb::flow_control& fc) -> PacketMatrix {
        PacketMatrix burst{};
        struct pcap_pkthdr* header;
        const u_char* pkt;
        size_t count = 0;

        while (count < burst_size) {
            int ret = pcap_next_ex(pcap, &header, &pkt);
            if (ret <= 0 || header->caplen > max_packet_size) continue;
            std::memcpy(burst[count].data(), pkt, header->caplen);
            count++;
        }
        return burst;
    }};

    using ParsedMatrix = PacketMatrix;
    tbb::flow::function_node<PacketMatrix, ParsedMatrix> parser_node{
        g, tbb::flow::unlimited, [](PacketMatrix burst) -> ParsedMatrix {
            ParsedMatrix parsed{};
            size_t idx = 0;
            for (const auto& pkt : burst) {
                if (is_ipv4(pkt.data())) {
                    parsed[idx++] = pkt;
                }
                else if (is_ipv6(pkt.data())) {
                    parsed[idx++] = pkt;
                }

            }
            return parsed;
        }
    };

    tbb::flow::function_node<ParsedMatrix, ParsedMatrix> routing_node{
        g, tbb::flow::unlimited, [&](ParsedMatrix packets) -> ParsedMatrix {
            sycl::buffer<uint8_t, 2> buf((uint8_t*)packets.data(), sycl::range<2>(burst_size, packet_fields));

            auto event = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(burst_size), [=](sycl::id<1> i) {
                    if (acc[i][12] == 0x08 && acc[i][13] == 0x00) {
                        for (int j = 0; j < 4; ++j) acc[i][30 + j] += 1;
                    }
                });
            });

            event.wait();
            auto start_ns = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end_ns = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            double duration_ms = static_cast<double>(end_ns - start_ns) / 1e6;
            std::cout << "[Profiling] Routing kernel execution time: " << duration_ms << " ms\n";

            return packets;
        }
    };

    tbb::flow::function_node<ParsedMatrix> send_node{
        g, tbb::flow::serial, [interface_name](const ParsedMatrix& packets) {
            int sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
            if (sock == -1) {
                perror("Socket creation failed");
                return;
            }

            struct sockaddr_ll device = {};
            device.sll_family = AF_PACKET;
            device.sll_protocol = htons(ETH_P_ALL);
            device.sll_ifindex = if_nametoindex(interface_name);

            for (const auto& pkt : packets) {
            	sendto(sock, pkt.data(), max_packet_size, 0, (struct sockaddr*)&device, sizeof(device));
            }
            close(sock);
        }
    };

    tbb::flow::make_edge(input_node, parser_node);
    tbb::flow::make_edge(parser_node, routing_node);
    tbb::flow::make_edge(routing_node, send_node);

    input_node.activate();
    g.wait_for_all();

    std::cout << "Pipeline complete.\n";
    pcap_close(pcap);
    return 0;
}

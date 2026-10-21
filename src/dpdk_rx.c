#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>
#include <rte_errno.h>

#define FRAME_COUNT 16
#define FRAME_BYTES (224 * 224 * 3)
#define CHUNK_BYTES 1400
#define SEGMENTS ((FRAME_BYTES + CHUNK_BYTES - 1) / CHUNK_BYTES)
#define HEADER_BYTES 22
#define MAGIC 0x45453534u
#define UDP_PORT 5000

static uint8_t frames[FRAME_COUNT][FRAME_BYTES];
static uint8_t seen[FRAME_COUNT][SEGMENTS];
static unsigned received_segments[FRAME_COUNT];

static uint64_t packets_received;
static uint64_t packets_dropped;
static uint64_t duplicate_chunks;
static uint64_t accepted_chunks;
static unsigned complete_frames;

static uint16_t read_be16(const uint8_t *p)
{
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint32_t read_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           p[3];
}

static void process_packet(const uint8_t *pkt, uint32_t size)
{
    if (size < 14 + 20 || read_be16(pkt + 12) != 0x0800)
        goto drop;

    const uint8_t *ip = pkt + 14;
    unsigned ihl = (ip[0] & 0x0f) * 4u;

    if ((ip[0] >> 4) != 4 || ihl < 20)
        goto drop;
    if (size < 14u + ihl + 8u)
        goto drop;

    unsigned ip_len = read_be16(ip + 2);
    if (ip_len < ihl + 8u || ip_len > size - 14u)
        goto drop;
    if (ip[9] != 17 || (read_be16(ip + 6) & 0x3fff))
        goto drop;

    const uint8_t *udp = ip + ihl;
    unsigned udp_len = read_be16(udp + 4);

    if (read_be16(udp + 2) != UDP_PORT)
        goto drop;
    if (udp_len < 8u + HEADER_BYTES || udp_len != ip_len - ihl)
        goto drop;

    const uint8_t *header = udp + 8;

    uint32_t magic = read_be32(header);
    uint32_t frame_id = read_be32(header + 4);
    unsigned seq = read_be16(header + 8);
    unsigned nseg = read_be16(header + 10);
    uint32_t offset = read_be32(header + 12);
    unsigned length = read_be16(header + 16);
    uint32_t frame_bytes = read_be32(header + 18);

    if (magic != MAGIC || frame_id >= FRAME_COUNT)
        goto drop;
    if (frame_bytes != FRAME_BYTES || nseg != SEGMENTS || seq >= nseg)
        goto drop;

    if (offset != seq * CHUNK_BYTES || offset >= FRAME_BYTES)
        goto drop;

    unsigned expected_length = FRAME_BYTES - offset;
    if (expected_length > CHUNK_BYTES)
        expected_length = CHUNK_BYTES;

    if (length != expected_length)
        goto drop;
    if (udp_len != 8u + HEADER_BYTES + length)
        goto drop;

    if (seen[frame_id][seq]) {
        duplicate_chunks++;
        return;
    }

    memcpy(frames[frame_id] + offset, header + HEADER_BYTES, length);
    seen[frame_id][seq] = 1;
    received_segments[frame_id]++;
    accepted_chunks++;

    if (received_segments[frame_id] == SEGMENTS) {
        complete_frames++;
        printf("Frame %u complete (%u/%u)\n",
               frame_id, complete_frames, FRAME_COUNT);
    }
    return;

drop:
    packets_dropped++;
}

static void check_result(int ret, const char *operation)
{
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "%s failed: %s\n",
                 operation, rte_strerror(-ret));
}

int main(int argc, char **argv)
{
    int used = rte_eal_init(argc, argv);
    if (used < 0)
        rte_exit(EXIT_FAILURE, "EAL initialization failed\n");

    argc -= used;
    argv += used;

    if (argc != 2)
        rte_exit(EXIT_FAILURE,
                 "Usage: dpdk_rx [EAL options] -- output.bin\n");

    const char *output_path = argv[1];
    uint16_t port;

    check_result(
        rte_eth_dev_get_port_by_name("net_pcap0", &port),
        "Find net_pcap0"
    );

    struct rte_mempool *pool = rte_pktmbuf_pool_create(
        "mbufs", 8191, 256, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id()
    );
    if (pool == NULL)
        rte_exit(EXIT_FAILURE, "Mempool failed: %s\n",
                 rte_strerror(rte_errno));

    struct rte_eth_conf conf = {0};
    check_result(
        rte_eth_dev_configure(port, 1, 1, &conf),
        "Configure port"
    );

    uint16_t rx_desc = 1024;
    uint16_t tx_desc = 1024;
    check_result(
        rte_eth_dev_adjust_nb_rx_tx_desc(port, &rx_desc, &tx_desc),
        "Adjust descriptors"
    );

    check_result(
        rte_eth_rx_queue_setup(
            port, 0, rx_desc, rte_socket_id(), NULL, pool
        ),
        "Set up RX"
    );
    check_result(
        rte_eth_tx_queue_setup(
            port, 0, tx_desc, rte_socket_id(), NULL
        ),
        "Set up TX"
    );
    check_result(rte_eth_dev_start(port), "Start port");

    const uint64_t timeout = 2 * rte_get_timer_hz();
    uint64_t last_packet = rte_get_timer_cycles();
    struct rte_mbuf *bufs[32];

    printf("Receiving from PCAP; idle timeout = 2 seconds\n");

    for (;;) {
        uint16_t n = rte_eth_rx_burst(port, 0, bufs, 32);

        if (n == 0) {
            if (rte_get_timer_cycles() - last_packet >= timeout)
                break;
            rte_delay_us_sleep(1000);
            continue;
        }

        last_packet = rte_get_timer_cycles();

        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf *m = bufs[i];
            packets_received++;

            if (rte_pktmbuf_linearize(m) != 0) {
                packets_dropped++;
            } else {
                process_packet(
                    rte_pktmbuf_mtod(m, const uint8_t *),
                    rte_pktmbuf_pkt_len(m)
                );
            }
            rte_pktmbuf_free(m);
        }
    }

    printf("\nPackets received: %" PRIu64 "\n", packets_received);
    printf("Accepted chunks: %" PRIu64 "\n", accepted_chunks);
    printf("Dropped invalid/unrelated packets: %" PRIu64 "\n",
           packets_dropped);
    printf("Duplicate chunks ignored: %" PRIu64 "\n", duplicate_chunks);
    printf("Complete frames: %u/%u\n", complete_frames, FRAME_COUNT);

    FILE *output = fopen(output_path, "wb");
    if (output == NULL)
        rte_exit(EXIT_FAILURE, "Cannot open output: %s\n", output_path);

    for (unsigned id = 0; id < FRAME_COUNT; id++) {
        if (received_segments[id] == SEGMENTS) {
            if (fwrite(frames[id], 1, FRAME_BYTES, output) != FRAME_BYTES)
                rte_exit(EXIT_FAILURE, "Output write failed\n");
        } else {
            fprintf(stderr, "Incomplete frame %u: %u/%u chunks\n",
                    id, received_segments[id], SEGMENTS);
        }
    }

    if (fclose(output) != 0)
        rte_exit(EXIT_FAILURE, "Output close failed\n");

    printf("Saved: %s\n", output_path);

    check_result(rte_eth_dev_stop(port), "Stop port");
    check_result(rte_eth_dev_close(port), "Close port");
    rte_mempool_free(pool);
    rte_eal_cleanup();

    return complete_frames == FRAME_COUNT ? EXIT_SUCCESS : EXIT_FAILURE;
}
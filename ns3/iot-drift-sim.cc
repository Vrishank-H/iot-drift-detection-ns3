// IoT network traffic with concept drift, for IF-DLDD drift detection.
// 30 WiFi sensor nodes -> 1 gateway (AP). Five concepts of T seconds each, mirroring
// Table 10 of Chu et al. (IEEE Access 2026):
//   C1 benign | C2 + brute force (22/23) | C3 + DoS Hulk/Slowloris (80)
//   C4 + small web attack (80) | C5 + botnet: port scan, DDoS flood (9), C&C (8080)
// The gateway logs every IP packet it sends/receives -> packets.csv (see process_traffic.py).
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>

using namespace ns3;

// One app for all traffic: uniform random gap in [0.5,1.5]*mean, uniform size,
// optional new source port every pktsPerFlow packets (new flow), optional port scan.
class IotTrafficApp : public Application
{
  public:
    void Setup(Ipv4Address dst, uint16_t port, double meanGap, uint32_t minSize, uint32_t maxSize,
               uint32_t pktsPerFlow, bool scan = false)
    {
        m_dst = dst;
        m_port = port;
        m_min = minSize;
        m_max = maxSize;
        m_ppf = pktsPerFlow;
        m_scan = scan;
        m_gap = CreateObject<UniformRandomVariable>();
        m_gap->SetAttribute("Min", DoubleValue(0.5 * meanGap));
        m_gap->SetAttribute("Max", DoubleValue(1.5 * meanGap));
        m_size = CreateObject<UniformRandomVariable>();
    }

  private:
    void StartApplication() override
    {
        NewSocket();
        Schedule();
    }

    void StopApplication() override
    {
        Simulator::Cancel(m_ev);
        m_sock->Close();
    }

    void NewSocket()
    {
        if (m_sock)
        {
            m_sock->Close();
        }
        m_sock = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_sock->Bind();
        m_sent = 0;
    }

    void Send()
    {
        if (m_ppf && m_sent == m_ppf)
        {
            NewSocket();
        }
        uint16_t port = m_scan ? 1024 + (m_next++ % 40000) : m_port;
        m_sock->SendTo(Create<Packet>(m_size->GetInteger(m_min, m_max)), 0,
                       InetSocketAddress(m_dst, port));
        m_sent++;
        Schedule();
    }

    void Schedule()
    {
        m_ev = Simulator::Schedule(Seconds(m_gap->GetValue()), &IotTrafficApp::Send, this);
    }

    Ptr<Socket> m_sock;
    Ipv4Address m_dst;
    uint16_t m_port{0};
    uint32_t m_min{0}, m_max{0}, m_ppf{0}, m_sent{0}, m_next{0};
    bool m_scan{false};
    Ptr<UniformRandomVariable> m_gap, m_size;
    EventId m_ev;
};

static std::ofstream g_log;

static void
LogPacket(Ptr<const Packet> pkt, Ptr<Ipv4>, uint32_t)
{
    Ptr<Packet> p = pkt->Copy();
    Ipv4Header ip;
    p->RemoveHeader(ip);
    uint16_t sp = 0, dp = 0;
    if (ip.GetProtocol() == UdpL4Protocol::PROT_NUMBER)
    {
        UdpHeader udp;
        p->PeekHeader(udp);
        sp = udp.GetSourcePort();
        dp = udp.GetDestinationPort();
    }
    g_log << Simulator::Now().GetSeconds() << ',' << ip.GetSource() << ',' << sp << ','
          << ip.GetDestination() << ',' << dp << ',' << +ip.GetProtocol() << ',' << pkt->GetSize()
          << '\n';
}

// Gateway service: answer every datagram with a fixed-size reply (CoAP ack, login fail, HTTP...).
// Datagrams to closed ports get ICMP port-unreachable from the ns-3 stack.
static void
Reply(uint32_t size, Ptr<Socket> s)
{
    Address from;
    while (s->RecvFrom(from))
    {
        s->SendTo(Create<Packet>(size), 0, from);
    }
}

int
main(int argc, char* argv[])
{
    uint32_t nNodes = 30;
    double T = 100; // seconds per concept
    uint32_t run = 1;
    std::string out = "packets.csv";
    bool pcap = false;
    CommandLine cmd(__FILE__);
    cmd.AddValue("nNodes", "Number of IoT sensor nodes (>= 25)", nNodes);
    cmd.AddValue("T", "Seconds per concept (5 concepts)", T);
    cmd.AddValue("run", "RNG run number (different traffic realisation)", run);
    cmd.AddValue("out", "Packet log CSV", out);
    cmd.AddValue("pcap", "Also write gateway pcap", pcap);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(nNodes < 25, "nNodes must be >= 25 (attack roles use nodes 0..24)");
    RngSeedManager::SetRun(run);

    NodeContainer gw(1), sensors(nNodes);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    Ssid ssid("iot-net");
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer devs = wifi.Install(phy, mac, gw);
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    devs.Add(wifi.Install(phy, mac, sensors));

    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(gw);
    mob.SetPositionAllocator("ns3::RandomDiscPositionAllocator",
                             "Rho", StringValue("ns3::UniformRandomVariable[Min=2|Max=25]"));
    mob.Install(sensors);

    InternetStackHelper().Install(NodeContainer(gw, sensors));
    Ipv4AddressHelper addr("10.1.1.0", "255.255.255.0");
    addr.Assign(devs); // gateway = 10.1.1.1, sensor i = 10.1.1.(i+2)
    Ipv4Address gwAddr("10.1.1.1");

    for (auto [port, size] : std::vector<std::pair<uint16_t, uint32_t>>{
             {5683, 16}, {22, 40}, {23, 40}, {80, 300}, {8080, 64}})
    {
        Ptr<Socket> s = Socket::CreateSocket(gw.Get(0), UdpSocketFactory::GetTypeId());
        s->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
        s->SetRecvCallback(MakeBoundCallback(&Reply, size));
    }

    auto add = [&](uint32_t node, double start, double stop, uint16_t port, double gap,
                   uint32_t lo, uint32_t hi, uint32_t ppf, bool scan = false) {
        Ptr<IotTrafficApp> app = CreateObject<IotTrafficApp>();
        app->Setup(gwAddr, port, gap, lo, hi, ppf, scan);
        sensors.Get(node)->AddApplication(app);
        app->SetStartTime(Seconds(start));
        app->SetStopTime(Seconds(stop));
    };

    double end = 5 * T;
    // Benign telemetry (CoAP-like, port 5683) on every node for the whole run:
    // temperature sensors 1 s, smart meters 2 s, cameras 0.25 s.
    for (uint32_t i = 0; i < nNodes; i++)
    {
        if (i % 10 < 7)
        {
            add(i, 1, end, 5683, 1.0, 40, 80, 0);
        }
        else if (i % 10 < 9)
        {
            add(i, 1, end, 5683, 2.0, 150, 250, 0);
        }
        else
        {
            add(i, 1, end, 5683, 0.25, 500, 900, 0);
        }
    }
    // Attacks from compromised sensors (they keep sending benign telemetry too).
    for (uint32_t i = 0; i < 6; i++) // C2: SSH/Telnet brute force, 3 tries per connection
    {
        add(i, T, 2 * T, i < 3 ? 22 : 23, 0.05, 60, 100, 3);
    }
    for (uint32_t i = 6; i < 14; i++) // C3: DoS Hulk, new connection every 15 requests
    {
        add(i, 2 * T, 3 * T, 80, 0.016, 800, 1400, 15);
    }
    for (uint32_t i = 14; i < 16; i++) // C3: Slowloris, long trickling connection
    {
        add(i, 2 * T, 3 * T, 80, 2.0, 10, 30, 0);
    }
    for (uint32_t i = 16; i < 18; i++) // C4: web attack (SQLi/XSS payloads)
    {
        add(i, 3 * T, 4 * T, 80, 0.25, 300, 700, 2);
    }
    for (uint32_t i = 18; i < 21; i++) // C5: port scan, one probe per flow
    {
        add(i, 4 * T, end, 0, 0.06, 0, 0, 1, true);
    }
    for (uint32_t i = 0; i < 12; i++) // C5: DDoS UDP flood to discard port
    {
        add(i, 4 * T, end, 9, 0.02, 1000, 1400, 25);
    }
    for (uint32_t i = 21; i < 25; i++) // C5: bot C&C beacons
    {
        add(i, 4 * T, end, 8080, 2.0, 30, 60, 1);
    }

    g_log.open(out);
    g_log << "t,src,sport,dst,dport,proto,bytes\n";
    Config::ConnectWithoutContext("/NodeList/0/$ns3::Ipv4L3Protocol/Rx", MakeCallback(&LogPacket));
    Config::ConnectWithoutContext("/NodeList/0/$ns3::Ipv4L3Protocol/Tx", MakeCallback(&LogPacket));
    if (pcap)
    {
        phy.EnablePcap("iot-gateway", devs.Get(0));
    }

    const char* concepts[] = {"C1 benign telemetry", "C2 + SSH/Telnet brute force",
                              "C3 + DoS Hulk / Slowloris", "C4 + web attack",
                              "C5 + botnet: port scan, DDoS, C&C"};
    for (int c = 0; c < 5; c++)
    {
        Simulator::Schedule(Seconds(c * T), [c, &concepts] {
            std::cout << "t=" << Simulator::Now().GetSeconds() << "s  concept " << concepts[c]
                      << std::endl;
        });
    }
    Simulator::Stop(Seconds(end + 1));
    Simulator::Run();
    Simulator::Destroy();
    g_log.close();
    std::cout << "Simulated " << end << " s, " << nNodes << " IoT nodes -> " << out << std::endl;
    return 0;
}

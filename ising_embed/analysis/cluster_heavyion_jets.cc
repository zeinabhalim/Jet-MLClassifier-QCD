#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

using namespace std;
using namespace fastjet;

// Filter: only charged stable particles
bool is_experimental_candidate(int pdg) {
    int apdg = abs(pdg);
    if (apdg == 211)  return true;   // pi+/- (dominant)
    if (apdg == 321)  return true;   // K+/-
    if (apdg == 2212) return true;   // p/pbar
    if (apdg == 11)   return true;   // e+/- 
    if (apdg == 13)   return true;   // mu+/-
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <input.oscar> <output.csv> <label>" << endl;
        cerr << "  label: 1=signal (Ising-embedded), 0=baseline" << endl;
        return 1;
    }

    string input_path  = argv[1];
    string output_path = argv[2];
    int    label       = atoi(argv[3]);

    double R             = 0.4;
    double pt_min        = 5.0;
    double pt_min_const  = 0.15;
    bool   use_filter    = (label >= 0);  // Filter for charged particles

    JetDefinition jet_def(antikt_algorithm, R);

    ifstream file(input_path);
    if (!file.is_open()) {
        cerr << "Error: cannot open " << input_path << endl;
        return 1;
    }

    ofstream out(output_path);
    out << "label,event,jet_pt,jet_eta,jet_phi,jet_mass,"
        << "nconstituents,pt_d,sqrt_d12" << endl;

    int event_num = 0, jet_count = 0;
    string line;

    // ── Read events ──────────────────────────────────────────────
    while (file.peek() != EOF) {
        // Find next event header
        bool found = false;
        while (getline(file, line)) {
            if (line.find("event") != string::npos && line.find("end") == string::npos) {
                found = true;
                break;
            }
        }
        if (!found) break;

        // Read particles until "event N end" or blank line
        vector<PseudoJet> parts;
        while (getline(file, line)) {
            if (line.empty()) continue;
            if (line.find("end") != string::npos) break;
            if (line[0] == '#') continue;

            istringstream iss(line);
            double t, x, y, z, m, p0, px, py, pz;
            int pdg, id, ch;
            if (!(iss >> t >> x >> y >> z >> m >> p0 >> px >> py >> pz >> pdg >> id >> ch))
                continue;

            double pt = sqrt(px * px + py * py);
            if (pt < pt_min_const) continue;

            // Apply charged-particle filter for experimental realism
            if (use_filter && !is_experimental_candidate(pdg)) continue;

            parts.push_back(PseudoJet(px, py, pz, p0));
        }

        if (parts.size() < 2) { event_num++; continue; }

        // ── Cluster jets ─────────────────────────────────────────
        ClusterSequence cs(parts, jet_def);
        vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(pt_min));

        for (auto& jet : jets) {
            if (jet.pt() < pt_min) continue;

            vector<PseudoJet> consts = jet.constituents();
            int nconst = (int)consts.size();

            // pT dispersion: sqrt(sum pTi^2) / sum pTi
            double sum_pt = 0.0, sum_pt2 = 0.0;
            for (auto& c : consts) {
                double cpt = c.pt();
                sum_pt  += cpt;
                sum_pt2 += cpt * cpt;
            }
            double pt_d = (sum_pt > 0) ? sqrt(sum_pt2) / sum_pt : 0.0;

            // Splitting scale sqrt(d12) from Cambridge/Aachen reclustering
            double sqrt_d12 = 0.0;
            if (nconst >= 2) {
                JetDefinition ca_def(cambridge_algorithm, R);
                ClusterSequence csca(consts, ca_def);
                vector<PseudoJet> ca_jets = csca.exclusive_jets_up_to(2);
                if (ca_jets.size() == 2)
                    sqrt_d12 = sqrt(ca_jets[0].kt_distance(ca_jets[1]));
            }

            out << label << ","
                << event_num << ","
                << jet.pt() << ","
                << jet.eta() << ","
                << jet.phi() << ","
                << max(jet.m(), 0.0) << ","
                << nconst << ","
                << pt_d << ","
                << sqrt_d12 << endl;
            jet_count++;
        }

        event_num++;
        if (event_num % 200 == 0)
            cout << "  Event " << event_num << " -> " << jet_count << " jets" << endl;
    }

    file.close();
    out.close();
    cout << "Done: " << event_num << " events, "
         << jet_count << " jets -> " << output_path << endl;
    return 0;
}


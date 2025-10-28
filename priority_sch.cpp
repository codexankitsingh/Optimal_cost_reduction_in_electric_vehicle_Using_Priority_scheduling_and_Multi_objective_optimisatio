#include <bits/stdc++.h>
#include<iostream>
using namespace std;

/*
  Stage-I: Priority Scheduling (C++)
  Implements Eq. (7), (9), (10), (22)-(26) from the paper and reproduces the
  "Numerical example (Stage-I)".

  - No lambda expressions used anywhere.
  - Modular functions for each equation / factor.
  - Outputs intermediate values and final assignment to match the paper's numbers.
*/

/* -----------------------
   Data structures
   ----------------------- */
struct EV {
    int id;
    double Ecap;     // kWh
    double SoC_init; // fraction
    double SoC_max;  // fraction
    double R_i;      // kW (charger rating)
    double T_stay;   // hours
};

struct Weights {
    double w_s;
    double w_d;
    double w_g;
    double w_p;
};

struct EVDetails {
    int id;
    double DeltaE;
    double p_req;
    double phi;
    double Dfactor;
    double Gfactor;
    double Pfactor;
    double lambda;
};

/* -----------------------
   Equation implementations (modular)
   ----------------------- */

// Eq. (7): DeltaE_i = (SoC_max - SoC_init) * Ecap
double calcDeltaE(const EV& ev) {
    double diff = ev.SoC_max - ev.SoC_init;
    return (diff > 0.0) ? diff * ev.Ecap : 0.0;
}

// Eq. (9) with cap at R_i to match the paper's numerical example:
// p_req = DeltaE / T_stay ; but practical p_req used in example is capped at R_i
double calcPReq(const EV& ev) {
    double dE = calcDeltaE(ev);
    if (ev.T_stay <= 0.0) {
        return (dE > 0.0) ? numeric_limits<double>::infinity() : 0.0;
    }
    double raw = dE / ev.T_stay;
    // cap at charger rating R_i to reproduce example numbers (implied in paper)
    return std::min(raw, ev.R_i);
}

// Eq. (10): phi = min(1, p_req / P_ref) ; P_ref is taken as R_i (charger rating)
double calcPhi(const EV& ev) {
    double p_req = calcPReq(ev);
    double pref = ev.R_i;
    if (pref <= 0.0) return 0.0;
    return std::min(1.0, p_req / pref);
}

// Eq. (23): D_factor = normalized (cdeg * DeltaE) using min-max normalization
vector<double> calcDegradationFactors(const vector<EV>& evs, double cdeg) {
    size_t n = evs.size();
    vector<double> raw(n);
    for (size_t i = 0; i < n; ++i) raw[i] = cdeg * calcDeltaE(evs[i]);
    double vmin = *min_element(raw.begin(), raw.end());
    double vmax = *max_element(raw.begin(), raw.end());
    double denom = (vmax - vmin);
    if (denom == 0.0) denom = 1.0; // guard against zero division
    vector<double> D(n);
    for (size_t i = 0; i < n; ++i) {
        D[i] = (raw[i] - vmin) / denom;
    }
    return D;
}

// Eq. (24): G_t_factor = max(0, (P_hat_agg - P_avg) / (P_max - P_avg))
// where P_hat_agg = sum_j p_req_j (we use capped p_req as in example)
double calcGridStressFactor(const vector<EV>& evs, double P_avg, double P_max) {
    double p_hat = 0.0;
    for (const EV& ev : evs) p_hat += calcPReq(ev);
    double denom = (P_max - P_avg);
    if (denom == 0.0) denom = 1.0; // guard
    double val = (p_hat - P_avg) / denom;
    return std::max(0.0, val);
}

// Eq. (25)-(26): normalize buy/rev prices and combine
double calcPriceFactor(double pi_buy, double pi_rev,
                       double pi_buy_min, double pi_buy_max,
                       double pi_rev_min, double pi_rev_max) {
    double buy_denom = (pi_buy_max - pi_buy_min);
    double rev_denom = (pi_rev_max - pi_rev_min);
    if (buy_denom == 0.0) buy_denom = 1.0;
    if (rev_denom == 0.0) rev_denom = 1.0;
    double P_buy = (pi_buy - pi_buy_min) / buy_denom;
    double P_rev = (pi_rev - pi_rev_min) / rev_denom;
    return P_buy - P_rev;
}

// Eq. (22): lambda_i = w_s*phi_i - w_d*D_i - w_g*G_t - w_p*P_t
vector<EVDetails> calcPriorityScores(const vector<EV>& evs,
                                     const Weights& weights,
                                     double cdeg,
                                     double P_avg, double P_max,
                                     double pi_buy, double pi_rev,
                                     double pi_buy_min, double pi_buy_max,
                                     double pi_rev_min, double pi_rev_max) {
    size_t n = evs.size();
    vector<EVDetails> details(n);

    // compute phi and p_req and DeltaE
    for (size_t i = 0; i < n; ++i) {
        details[i].id = evs[i].id;
        details[i].DeltaE = calcDeltaE(evs[i]);
        details[i].p_req = calcPReq(evs[i]);
        details[i].phi = calcPhi(evs[i]);
    }

    // degradation normalized factors
    vector<double> D = calcDegradationFactors(evs, cdeg);
    for (size_t i = 0; i < n; ++i) details[i].Dfactor = D[i];

    // system-level factors
    double Gfactor = calcGridStressFactor(evs, P_avg, P_max);
    double Pfactor = calcPriceFactor(pi_buy, pi_rev,
                                     pi_buy_min, pi_buy_max,
                                     pi_rev_min, pi_rev_max);

    for (size_t i = 0; i < n; ++i) {
        details[i].Gfactor = Gfactor;
        details[i].Pfactor = Pfactor;
        // Eq. (22)
        details[i].lambda = weights.w_s * details[i].phi
                          - weights.w_d * details[i].Dfactor
                          - weights.w_g * details[i].Gfactor
                          - weights.w_p * details[i].Pfactor;
    }

    return details;
}

/* Comparator functor (no lambdas used)
   Sort pairs (id, score) descending by score; tie-breaker: smaller id first.
*/
struct ScoreComparator {
    bool operator()(const pair<int,double>& a, const pair<int,double>& b) const {
        if (a.second != b.second) return a.second > b.second; // higher score first
        return a.first < b.first; // lower id first if tie
    }
};

// Assignment: pick top M by descending lambda
vector<int> assignChargers(const vector<EVDetails>& details, int M) {
    vector<pair<int,double>> id_score;
    for (const EVDetails& d : details) id_score.push_back(make_pair(d.id, d.lambda));
    sort(id_score.begin(), id_score.end(), ScoreComparator()); // uses functor
    vector<int> admitted;
    for (int i = 0; i < M && i < (int)id_score.size(); ++i) admitted.push_back(id_score[i].first);
    return admitted;
}

/* -----------------------
   Main: numerical example from paper
   ----------------------- */
int main() {
    // System params (paper)
    int N = 3;
    int M = 2;
    double P_max = 25.0;   // kW
    double P_avg = 12.0;   // kW
    double cdeg = 0.02;    // $/kWh
    double pi_buy = 0.25;
    double pi_rev = 0.18;
    double pi_buy_min = 0.10, pi_buy_max = 0.50;
    double pi_rev_min = 0.05, pi_rev_max = 0.30;
    Weights w = {0.25, 0.25, 0.25, 0.25};

    // EVs: Table I
    vector<EV> evs = {
        {1, 60.0, 0.30, 0.80, 7.0, 4.0},
        {2, 40.0, 0.60, 0.90, 11.0, 1.0},
        {3, 50.0, 0.20, 0.60, 7.0, 6.0}
    };

    // compute
    vector<EVDetails> details = calcPriorityScores(evs, w, cdeg,
                                                   P_avg, P_max,
                                                   pi_buy, pi_rev,
                                                   pi_buy_min, pi_buy_max,
                                                   pi_rev_min, pi_rev_max);

    // Output formatting
    cout.setf(ios::fixed);
    cout << setprecision(3);

    cout << "=== Stage-I Priority Scheduling (Numerical Example) ===\n\n";

    // Step 1 — Urgency (DeltaE, p_req, phi)
    cout << "Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):\n";
    for (const EVDetails& d : details) {
        cout << "EV" << d.id
             << ": DeltaE = " << setw(6) << d.DeltaE << " kWh"
             << ", p_req = " << setw(6) << d.p_req << " kW"
             << ", phi = " << setw(6) << d.phi << '\n';
    }
    cout << '\n';

    // Step 2 — Degradation
    cout << "Step 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):\n";
    for (const EVDetails& d : details) {
        double raw_deg = cdeg * d.DeltaE;
        cout << "EV" << d.id
             << ": cdeg*DeltaE = " << setw(6) << raw_deg
             << ", Dfactor = " << setw(6) << d.Dfactor << '\n';
    }
    cout << '\n';

    // Grid stress and price
    double Gfactor = details.front().Gfactor; // same for all
    double Pfactor = details.front().Pfactor;
    cout << "Step 3 — Grid stress factor (Gfactor) (Eq. (24)):\n";
    cout << "Gfactor = " << setprecision(4) << Gfactor << '\n' << '\n';

    cout << "Step 4 — Price factor (Pfactor) (Eq. (25),(26)):\n";
    cout << setprecision(3) << "Pfactor = " << Pfactor << '\n' << '\n';

    // Lambdas
    cout << "Step 5 — Priority scores (lambda_i) (Eq. (22)):\n";
    cout << setprecision(5);
    for (const EVDetails& d : details) {
        cout << "lambda_" << d.id << " = " << setw(8) << d.lambda << '\n';
    }
    cout << '\n';

    // Ranking & assignment
    cout << "Step 6 — Ranking and Admission:\n";
    // build vector of pairs and sort (we'll reuse the same comparator functor)
    vector<pair<int,double>> id_score;
    for (const EVDetails& d : details) id_score.push_back(make_pair(d.id, d.lambda));
    sort(id_score.begin(), id_score.end(), ScoreComparator());

    cout << "Ranking (desc lambda): ";
    for (size_t i = 0; i < id_score.size(); ++i) {
        if (i) cout << " > ";
        cout << "EV" << id_score[i].first;
    }
    cout << '\n';

    vector<int> admitted = assignChargers(details, M);
    cout << "With M=" << M << " chargers, admitted EVs: ";
    for (size_t i = 0; i < admitted.size(); ++i) {
        if (i) cout << ", ";
        cout << "EV" << admitted[i];
    }
    cout << '\n';
    cout << "Waiting EVs: ";
    bool first = true;
    for (const EV& ev : evs) {
        if (find(admitted.begin(), admitted.end(), ev.id) == admitted.end()) {
            if (!first) cout << ", ";
            cout << "EV" << ev.id;
            first = false;
        }
    }
    cout << '\n';

    return 0;
}

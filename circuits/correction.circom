pragma circom 2.1.6;

template CorrectionSlice(N) {
    signal input S_retain[N];
    signal input S_j[N];
    signal input dW_j[N];
    signal input dW_agg[N];
    signal input ETA;
    signal input dW_out[N];

    signal rhs_term1[N];
    signal etaS[N];
    signal rhs_term2[N];

    for (var i = 0; i < N; i++) {
        rhs_term1[i] <== dW_agg[i] * S_retain[i];
        etaS[i] <== ETA * S_j[i];
        rhs_term2[i] <== etaS[i] * dW_j[i];

        dW_out[i] === rhs_term1[i] - rhs_term2[i];
    }
}

component main {public [dW_agg, ETA, dW_out]} = CorrectionSlice(7168);

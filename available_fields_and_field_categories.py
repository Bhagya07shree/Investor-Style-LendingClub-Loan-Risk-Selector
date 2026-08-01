# =========================================================
# FIELD DEFINITIONS — human-meaningful raw fields only,
# NOT the 143 model-encoded features. preprocess.py handles
# turning these into what the model actually needs.
# =========================================================
AVAILABLE_FIELDS = {
    # --- Loan basics ---
    "loan_amnt": {"label": "Loan Amount ($)", "type": "number", "min": 1000, "max": 40000, "default": 10000},
    "term": {"label": "Loan Term", "type": "select", "options": ["36 months", "60 months"]},
    "sub_grade": {"label": "Loan Sub-Grade", "type": "select", "options": [f"{l}{n}" for l in "ABCDEFG" for n in range(1, 6)]},
    "purpose": {"label": "Loan Purpose", "type": "select", "options": ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "small_business", "medical", "car", "moving", "vacation", "house", "wedding", "renewable_energy"]},
    "initial_list_status": {"label": "Initial Listing Status", "type": "select", "options": ["f", "w"]},
    "int_rate": {"label": "Interest Rate (%)", "type": "number", "min": 5.0, "max": 31.0, "default": 13.0},
    "installment": {"label": "Monthly Installment ($)", "type": "number", "min": 20, "max": 2000, "default": 350},
    "tot_coll_amt": {"label": "Total Collections Amount ($)", "type": "number", "min": 0, "max": 100000, "default": 0},

    # --- Borrower income & employment ---
    "annual_inc": {"label": "Annual Income ($)", "type": "number", "min": 0, "max": 1000000, "default": 60000},
    "emp_length": {"label": "Employment Length", "type": "select", "options": ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]},
    "verification_status": {"label": "Income Verification Status", "type": "select", "options": ["Not Verified", "Source Verified", "Verified"]},
    "home_ownership": {"label": "Home Ownership", "type": "select", "options": ["MORTGAGE", "RENT", "OWN", "OTHER"]},
    "addr_state": {"label": "State", "type": "select", "options": ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]},

    # --- Debt & credit usage ---
    "dti": {"label": "Debt-to-Income Ratio (%)", "type": "number", "min": 0, "max": 65, "default": 18},
    "revol_bal": {"label": "Revolving Balance ($)", "type": "number", "min": 0, "max": 250000, "default": 15000},
    "revol_util": {"label": "Revolving Utilization (%)", "type": "number", "min": 0, "max": 150, "default": 50},
    "total_rev_hi_lim": {"label": "Total Revolving Credit Limit ($)", "type": "number", "min": 0, "max": 500000, "default": 30000},

    # --- Credit history ---
    "fico_range_low": {"label": "FICO Score (Low)", "type": "number", "min": 300, "max": 850, "default": 680},
    "fico_range_high": {"label": "FICO Score (High)", "type": "number", "min": 300, "max": 850, "default": 684},
    "earliest_cr_line": {"label": "Credit History Start Year", "type": "number", "min": 1950, "max": 2018, "default": 2000},
    "open_acc": {"label": "Open Credit Lines", "type": "number", "min": 0, "max": 60, "default": 10},
    "total_acc": {"label": "Total Credit Lines Ever Opened", "type": "number", "min": 0, "max": 120, "default": 20},
    "pub_rec": {"label": "Public Records", "type": "number", "min": 0, "max": 10, "default": 0},
    "pub_rec_bankruptcies": {"label": "Bankruptcies", "type": "number", "min": 0, "max": 5, "default": 0},
    "tax_liens": {"label": "Tax Liens", "type": "number", "min": 0, "max": 10, "default": 0},

    # --- Delinquency history ---
    "delinq_2yrs": {"label": "Delinquencies in Last 2 Years", "type": "number", "min": 0, "max": 20, "default": 0},
    "mths_since_last_delinq": {"label": "Months Since Last Delinquency", "type": "number_optional", "min": 0, "max": 240},
    "mths_since_last_record": {"label": "Months Since Last Public Record", "type": "number_optional", "min": 0, "max": 240},
    "inq_last_6mths": {"label": "Credit Inquiries in Last 6 Months", "type": "number", "min": 0, "max": 20, "default": 1},
    "acc_now_delinq": {"label": "Accounts Currently Delinquent", "type": "number", "min": 0, "max": 10, "default": 0},

    # --- Installment / mortgage accounts ---
    "mort_acc": {"label": "Mortgage Accounts", "type": "number", "min": 0, "max": 20, "default": 1},
    "tot_cur_bal": {"label": "Total Current Balance Across Accounts ($)", "type": "number", "min": 0, "max": 2000000, "default": 50000},
    "tot_hi_cred_lim": {"label": "Total High Credit Limit ($)", "type": "number", "min": 0, "max": 2000000, "default": 100000},
    "avg_cur_bal": {"label": "Average Current Balance per Account ($)", "type": "number", "min": 0, "max": 500000, "default": 10000},
    "acc_open_past_24mths": {"label": "Accounts Opened in Last 24 Months", "type": "number", "min": 0, "max": 25, "default": 3},

    # --- Recent activity ---
    "total_bal_ex_mort": {"label": "Total Balance Excluding Mortgage ($)", "type": "number", "min": 0, "max": 500000, "default": 30000},
    "total_bc_limit": {"label": "Total Bankcard Credit Limit ($)", "type": "number", "min": 0, "max": 250000, "default": 20000},
    "bc_open_to_buy": {"label": "Available Bankcard Credit ($)", "type": "number", "min": 0, "max": 250000, "default": 8000},
    "bc_util": {"label": "Bankcard Utilization (%)", "type": "number", "min": 0, "max": 150, "default": 60},
    "num_actv_bc_tl": {"label": "Active Bankcard Accounts", "type": "number", "min": 0, "max": 30, "default": 4},

    # --- Tradeline detail ---
    "num_bc_sats": {"label": "Satisfactory Bankcard Accounts", "type": "number", "min": 0, "max": 30, "default": 4},
    "num_bc_tl": {"label": "Total Bankcard Accounts", "type": "number", "min": 0, "max": 40, "default": 6},
    "num_il_tl": {"label": "Installment Accounts", "type": "number", "min": 0, "max": 40, "default": 5},
    "num_op_rev_tl": {"label": "Open Revolving Accounts", "type": "number", "min": 0, "max": 30, "default": 6},
    "num_rev_accts": {"label": "Total Revolving Accounts", "type": "number", "min": 0, "max": 60, "default": 10},
    "num_actv_rev_tl": {"label": "Active Revolving Accounts", "type": "number", "min": 0, "max": 30, "default": 5},
    "num_tl_op_past_12m": {"label": "Accounts Opened in Last 12 Months", "type": "number", "min": 0, "max": 15, "default": 1},
    "percent_bc_gt_75": {"label": "Bankcard Accounts Above 75% Utilization (%)", "type": "number", "min": 0, "max": 100, "default": 40},
    "pct_tl_nvr_dlq": {"label": "Accounts Never Delinquent (%)", "type": "number", "min": 0, "max": 100, "default": 94},

    # --- Severe delinquency indicators ---
    "num_tl_90g_dpd_24m": {"label": "Accounts 90+ Days Late in Last 24 Months", "type": "number", "min": 0, "max": 10, "default": 0},
    "num_tl_30dpd": {"label": "Accounts Currently 30 Days Late", "type": "number", "min": 0, "max": 5, "default": 0},
    "num_tl_120dpd_2m": {"label": "Accounts 120+ Days Late in Last 2 Months", "type": "number", "min": 0, "max": 5, "default": 0},
    "num_accts_ever_120_pd": {"label": "Accounts Ever 120+ Days Past Due", "type": "number", "min": 0, "max": 10, "default": 0},
    "collections_12_mths_ex_med": {"label": "Collections in Last 12 Months (Excl. Medical)", "type": "number", "min": 0, "max": 10, "default": 0},
    "chargeoff_within_12_mths": {"label": "Charge-Offs in Last 12 Months", "type": "number", "min": 0, "max": 5, "default": 0},

    # --- Account age / recency detail ---
    "mo_sin_old_rev_tl_op": {"label": "Months Since Oldest Revolving Account Opened", "type": "number", "min": 0, "max": 900, "default": 180},
    "mo_sin_old_il_acct": {"label": "Months Since Oldest Installment Account Opened", "type": "number", "min": 0, "max": 750, "default": 125},
    "mo_sin_rcnt_rev_tl_op": {"label": "Months Since Most Recent Revolving Account Opened", "type": "number", "min": 0, "max": 400, "default": 13},
    "mo_sin_rcnt_tl": {"label": "Months Since Most Recent Account Opened", "type": "number", "min": 0, "max": 250, "default": 8},
    "mths_since_recent_bc": {"label": "Months Since Most Recent Bankcard Opened", "type": "number", "min": 0, "max": 650, "default": 24},
    "mths_since_recent_bc_dlq": {"label": "Months Since Recent Bankcard Delinquency", "type": "number_optional", "min": 0, "max": 240},
    "mths_since_recent_inq": {"label": "Months Since Most Recent Credit Inquiry", "type": "number", "min": 0, "max": 36, "default": 6},
    "mths_since_recent_revol_delinq": {"label": "Months Since Recent Revolving Delinquency", "type": "number_optional", "min": 0, "max": 240},
    "mths_since_last_major_derog": {"label": "Months Since Major Negative Credit Event", "type": "number_optional", "min": 0, "max": 240},

    # --- Bureau tradeline fields ---
    "open_acc_6m": {"label": "Accounts Opened in Last 6 Months", "type": "number", "min": 0, "max": 12, "default": 1},
    "open_act_il": {"label": "Currently Open Installment Accounts", "type": "number", "min": 0, "max": 20, "default": 3},
    "open_il_12m": {"label": "Installment Accounts Opened in Last 12 Months", "type": "number", "min": 0, "max": 12, "default": 1},
    "open_il_24m": {"label": "Installment Accounts Opened in Last 24 Months", "type": "number", "min": 0, "max": 20, "default": 2},
    "mths_since_rcnt_il": {"label": "Months Since Most Recent Installment Account", "type": "number", "min": 0, "max": 350, "default": 21},
    "total_bal_il": {"label": "Total Installment Balance ($)", "type": "number", "min": 0, "max": 900000, "default": 35000},
    "il_util": {"label": "Installment Utilization (%)", "type": "number", "min": 0, "max": 150, "default": 70},
    "open_rv_12m": {"label": "Revolving Accounts Opened in Last 12 Months", "type": "number", "min": 0, "max": 12, "default": 1},
    "open_rv_24m": {"label": "Revolving Accounts Opened in Last 24 Months", "type": "number", "min": 0, "max": 20, "default": 3},
    "max_bal_bc": {"label": "Highest Bankcard Balance ($)", "type": "number", "min": 0, "max": 150000, "default": 5700},
    "all_util": {"label": "Overall Credit Utilization (%)", "type": "number", "min": 0, "max": 150, "default": 60},
    "inq_fi": {"label": "Finance Company Inquiries", "type": "number", "min": 0, "max": 15, "default": 1},
    "total_cu_tl": {"label": "Finance / Credit Union Accounts", "type": "number", "min": 0, "max": 30, "default": 1},
    "inq_last_12m": {"label": "Credit Inquiries in Last 12 Months", "type": "number", "min": 0, "max": 25, "default": 2},
    "total_il_high_credit_limit": {"label": "Total Installment Credit Limit ($)", "type": "number", "min": 0, "max": 1250000, "default": 40000},
}

FIELD_CATEGORIES = {
    "Loan basics": [
        "loan_amnt", "term", "sub_grade", "purpose", "initial_list_status",
        "int_rate", "installment", "tot_coll_amt",
    ],
    "Borrower income & employment": [
        "annual_inc", "emp_length", "verification_status",
        "home_ownership", "addr_state",
    ],
    "Debt & credit usage": [
        "dti", "revol_bal", "revol_util", "total_rev_hi_lim",
    ],
    "Credit history": [
        "fico_range_low", "fico_range_high", "earliest_cr_line",
        "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    ],
    "Delinquency history": [
        "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record",
        "inq_last_6mths", "acc_now_delinq",
    ],
    "Installment / mortgage accounts": [
        "mort_acc", "tot_cur_bal", "tot_hi_cred_lim",
        "avg_cur_bal", "acc_open_past_24mths",
    ],
    "Recent activity": [
        "total_bal_ex_mort", "total_bc_limit", "bc_open_to_buy",
        "bc_util", "num_actv_bc_tl",
    ],
    "Tradeline detail": [
        "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
        "num_rev_accts", "num_actv_rev_tl", "num_tl_op_past_12m",
        "percent_bc_gt_75", "pct_tl_nvr_dlq",
    ],
    "Severe delinquency indicators": [
        "num_tl_90g_dpd_24m", "num_tl_30dpd", "num_tl_120dpd_2m",
        "num_accts_ever_120_pd", "collections_12_mths_ex_med",
        "chargeoff_within_12_mths",
    ],
    "Account age / recency detail": [
        "mo_sin_old_rev_tl_op", "mo_sin_old_il_acct", "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
        "mths_since_recent_inq", "mths_since_recent_revol_delinq",
        "mths_since_last_major_derog",
    ],
    "Bureau tradeline fields": [
        "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
        "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m",
        "open_rv_24m", "max_bal_bc", "all_util", "inq_fi",
        "total_cu_tl", "inq_last_12m", "total_il_high_credit_limit",
    ],
}
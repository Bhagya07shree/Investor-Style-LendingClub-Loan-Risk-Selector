FEATURE_EXPLANATIONS = {

    # ===========================
    # Loan Information
    # ===========================

    "loan_amnt": {
        "display_name": "Loan Amount",
        "what": "The total amount of money the borrower is requesting.",
        "why": "The loan amount determines the size of the financial commitment and influences the monthly repayment burden.",
        "effect_on_prediction": "Larger loan amounts may be associated with higher default risk when they are not supported by sufficient income, strong credit history, or manageable debt levels.",
        "example": "A borrower requesting $35,000 generally carries a larger repayment obligation than someone requesting $5,000."
    },

    "term": {
        "display_name": "Loan Term",
        "what": "The length of time over which the borrower will repay the loan.",
        "why": "Longer repayment periods increase the time during which financial circumstances can change.",
        "effect_on_prediction": "Longer loan terms are often associated with slightly higher default risk because borrowers remain exposed to repayment obligations for a longer period.",
        "example": "A 60-month loan usually carries more long-term uncertainty than a 36-month loan."
    },

    "sub_grade": {
        "display_name": "Credit Sub-Grade",
        "what": "A detailed credit risk category assigned by Lending Club based on the borrower's profile.",
        "why": "It summarizes multiple aspects of creditworthiness into a finer risk classification.",
        "effect_on_prediction": "Lower-quality sub-grades generally correspond to higher predicted default risk because they reflect weaker overall credit profiles.",
        "example": "Sub-grade A1 typically represents stronger borrowers than sub-grade D5."
    },

    "purpose": {
        "display_name": "Loan Purpose",
        "what": "The primary reason the borrower is requesting the loan.",
        "why": "Borrowers requesting loans for different purposes may exhibit different historical repayment patterns.",
        "effect_on_prediction": "Certain loan purposes have historically shown different levels of default risk, making this feature useful for prediction.",
        "example": "A debt consolidation loan may have different repayment behavior compared to a small business loan."
    },

    "initial_list_status": {
        "display_name": "Initial Listing Status",
        "what": "The listing status assigned when the loan was first offered on the platform.",
        "why": "It reflects how the loan entered the lending marketplace.",
        "effect_on_prediction": "Although usually less influential than financial variables, it may capture subtle historical lending patterns.",
        "example": "Different listing categories may have slightly different historical default rates."
    },

    "int_rate": {
        "display_name": "Interest Rate",
        "what": "The annual interest rate charged on the loan.",
        "why": "Interest rate reflects the lender's assessment of borrower risk and determines the cost of borrowing.",
        "effect_on_prediction": "Higher interest rates are often associated with borrowers who were already considered riskier during loan approval.",
        "example": "A borrower paying 18% interest is generally considered riskier than one paying 8%."
    },

    "installment": {
        "display_name": "Monthly Installment",
        "what": "The fixed monthly payment required to repay the loan.",
        "why": "Monthly payment size affects the borrower's ability to comfortably meet repayment obligations.",
        "effect_on_prediction": "Higher installments may increase repayment pressure, particularly when combined with lower income or higher existing debt.",
        "example": "A monthly payment of $900 creates more financial pressure than one of $250."
    },

    "tot_coll_amt": {
        "display_name": "Total Collection Amount",
        "what": "The total amount that has been collected through collection agencies.",
        "why": "Collection activity may indicate previous financial difficulties or repayment issues.",
        "effect_on_prediction": "Higher collection amounts generally indicate a more troubled credit history and may increase predicted default risk.",
        "example": "A borrower with previous collection activity may represent higher credit risk."
    },

    # ===========================
    # Employment & Income
    # ===========================

    "annual_inc": {
        "display_name": "Annual Income",
        "what": "The borrower's total annual income before taxes.",
        "why": "Income represents the borrower's capacity to repay loan installments while managing other financial obligations.",
        "effect_on_prediction": "Higher income is generally associated with lower default risk, especially when considered alongside debt levels and loan size.",
        "example": "A borrower earning $120,000 per year generally has greater repayment capacity than someone earning $35,000."
    },

    "emp_length": {
        "display_name": "Employment Length",
        "what": "The number of years the borrower has been employed.",
        "why": "Long-term employment often indicates greater income stability and financial consistency.",
        "effect_on_prediction": "Borrowers with longer employment histories generally demonstrate lower repayment risk than those with limited employment history.",
        "example": "Someone employed for 10 years is often viewed as having more stable income than someone employed for only 6 months."
    },

    "verification_status": {
        "display_name": "Verification Status",
        "what": "Indicates whether the borrower's income information has been verified by the lender.",
        "why": "Verified financial information increases confidence in the accuracy of the loan application.",
        "effect_on_prediction": "Verified applications may demonstrate slightly different repayment patterns than unverified applications.",
        "example": "Income confirmed through official documents is generally considered more reliable than self-reported information."
    },

    "home_ownership": {
        "display_name": "Home Ownership",
        "what": "The borrower's housing status, such as owning, renting, or having a mortgage.",
        "why": "Housing status provides insight into financial stability and ongoing housing-related expenses.",
        "effect_on_prediction": "Different home ownership categories may exhibit different historical repayment behaviors.",
        "example": "A homeowner with an established mortgage may have a different financial profile than a renter."
    },

    "addr_state": {
        "display_name": "State of Residence",
        "what": "The U.S. state where the borrower resides.",
        "why": "Regional economic conditions and lending patterns can vary across states.",
        "effect_on_prediction": "Location may contribute modestly to the prediction by capturing historical regional trends.",
        "example": "Economic conditions in different states can influence borrower repayment behavior."
    },

    # ===========================
    # Debt & Credit Usage
    # ===========================

    "dti": {
        "display_name": "Debt-to-Income Ratio",
        "what": "The percentage of monthly income already committed to existing debt payments.",
        "why": "It measures how much of a borrower's income is already being used to repay debts.",
        "effect_on_prediction": "Higher DTI ratios generally indicate less financial flexibility and are often associated with increased default risk.",
        "example": "A borrower with a DTI of 20% has considerably more available income than someone with a DTI of 50%."
    },

    "revol_bal": {
        "display_name": "Revolving Balance",
        "what": "The total outstanding balance on revolving credit accounts, such as credit cards.",
        "why": "It reflects the amount of revolving debt the borrower currently carries.",
        "effect_on_prediction": "Higher revolving balances may indicate greater financial obligations, particularly when paired with high credit utilization.",
        "example": "Carrying $25,000 in revolving balances represents a larger debt burden than carrying $2,000."
    },

    "revol_util": {
        "display_name": "Revolving Credit Utilization",
        "what": "The percentage of available revolving credit currently being used.",
        "why": "Credit utilization is a widely used indicator of credit management and financial stress.",
        "effect_on_prediction": "Higher utilization levels have historically been associated with higher default rates.",
        "example": "Using 90% of available credit is generally considered riskier than using only 20%."
    },

    "total_rev_hi_lim": {
        "display_name": "Total Revolving Credit Limit",
        "what": "The total available credit limit across revolving accounts.",
        "why": "It represents the borrower's overall revolving credit capacity.",
        "effect_on_prediction": "Higher available credit can be a positive indicator when balances remain well controlled.",
        "example": "A borrower with a $50,000 credit limit and low balances generally appears stronger than one with a $5,000 limit."
    },

    # ===========================
    # Credit History
    # ===========================

    "fico_range_low": {
        "display_name": "FICO Score (Lower Range)",
        "what": "The lower value of the borrower's reported FICO credit score range.",
        "why": "FICO scores summarize a borrower's credit history and repayment behavior.",
        "effect_on_prediction": "Higher FICO scores are strongly associated with lower default risk.",
        "example": "A FICO score of 780 generally represents lower credit risk than a score of 620."
    },

    "fico_range_high": {
        "display_name": "FICO Score (Upper Range)",
        "what": "The upper value of the reported FICO score range.",
        "why": "It provides additional precision about the borrower's creditworthiness.",
        "effect_on_prediction": "Higher values generally indicate stronger historical credit performance.",
        "example": "A score range of 760–764 is generally stronger than 660–664."
    },

    "earliest_cr_line": {
        "display_name": "Earliest Credit Line",
        "what": "The year in which the borrower opened their first credit account.",
        "why": "A longer credit history provides more information about long-term repayment behavior.",
        "effect_on_prediction": "Borrowers with longer-established credit histories are often associated with lower uncertainty and lower risk.",
        "example": "Someone with a 20-year credit history generally has more established repayment records than someone with only 2 years."
    },

    "open_acc": {
        "display_name": "Open Credit Accounts",
        "what": "The number of credit accounts that are currently open.",
        "why": "It reflects the number of active financial obligations being managed.",
        "effect_on_prediction": "Both very low and very high numbers of open accounts may provide useful information depending on the overall credit profile.",
        "example": "Managing 8 open accounts responsibly may indicate stronger experience than having only one account."
    },

    "total_acc": {
        "display_name": "Total Credit Accounts",
        "what": "The total number of credit accounts the borrower has ever opened.",
        "why": "It represents the borrower's overall credit experience.",
        "effect_on_prediction": "A longer credit history with many well-managed accounts may contribute positively, although the effect depends on overall credit behavior.",
        "example": "A borrower with 25 lifetime accounts has a more extensive credit history than someone with only 3."
    },

    "pub_rec": {
        "display_name": "Public Records",
        "what": "The number of public records associated with the borrower's credit file.",
        "why": "Public records may indicate significant legal or financial events.",
        "effect_on_prediction": "Higher numbers of public records are generally associated with increased credit risk.",
        "example": "A borrower with multiple public records may have experienced serious financial difficulties."
    },

    "pub_rec_bankruptcies": {
        "display_name": "Bankruptcies",
        "what": "The number of bankruptcy records associated with the borrower.",
        "why": "Bankruptcy is considered one of the strongest indicators of previous financial distress.",
        "effect_on_prediction": "Borrowers with previous bankruptcies generally exhibit higher predicted default risk.",
        "example": "A borrower with one recorded bankruptcy is typically considered riskier than one with none."
    },

    "tax_liens": {
        "display_name": "Tax Liens",
        "what": "The number of tax liens recorded against the borrower.",
        "why": "Tax liens indicate unpaid tax obligations and may reflect financial challenges.",
        "effect_on_prediction": "Higher numbers of tax liens generally increase predicted default risk.",
        "example": "Unresolved tax liens may signal ongoing financial stress."
    },

    # ===========================
    # Delinquency History
    # ===========================

    "delinq_2yrs": {
        "display_name": "Delinquencies (Last 2 Years)",
        "what": "The number of times the borrower has been delinquent on credit payments during the past two years.",
        "why": "Recent missed payments are important indicators of repayment behavior.",
        "effect_on_prediction": "More recent delinquencies generally correspond to higher predicted default risk.",
        "example": "A borrower with three recent delinquencies typically represents higher risk than one with none."
    },

    "mths_since_last_delinq": {
        "display_name": "Months Since Last Delinquency",
        "what": "The number of months since the borrower's most recent delinquent payment.",
        "why": "It measures how recently repayment problems occurred.",
        "effect_on_prediction": "More recent delinquencies generally carry greater risk than delinquencies that occurred many years ago.",
        "example": "A delinquency occurring 6 months ago is typically more concerning than one occurring 8 years ago."
    },

    "mths_since_last_record": {
        "display_name": "Months Since Last Public Record",
        "what": "The number of months since the borrower's last recorded public record.",
        "why": "Recent public records may indicate ongoing financial difficulties.",
        "effect_on_prediction": "Recent public records generally contribute more strongly to predicted risk than older records.",
        "example": "A public record from last year usually has greater relevance than one recorded many years ago."
    },

    "inq_last_6mths": {
        "display_name": "Credit Inquiries (Last 6 Months)",
        "what": "The number of credit inquiries made during the previous six months.",
        "why": "Frequent credit applications may indicate increased borrowing activity.",
        "effect_on_prediction": "Higher inquiry counts are often associated with increased default risk, particularly when combined with other risk factors.",
        "example": "Six recent credit inquiries may indicate more active credit seeking than a single inquiry."
    },

    "acc_now_delinq": {
        "display_name": "Currently Delinquent Accounts",
        "what": "The number of accounts that are currently delinquent.",
        "why": "Current missed payments provide direct evidence of ongoing repayment difficulties.",
        "effect_on_prediction": "Borrowers with currently delinquent accounts generally have substantially higher predicted default risk.",
        "example": "An account currently overdue is a stronger warning sign than one that was resolved years ago."
    },

        # ===========================
    # Mortgage & Balance Features
    # ===========================

    "mort_acc": {
        "display_name": "Mortgage Accounts",
        "what": "The number of mortgage accounts the borrower currently has.",
        "why": "Mortgage accounts provide insight into the borrower's experience managing long-term debt.",
        "effect_on_prediction": "The impact depends on the overall financial profile. Well-managed mortgage accounts may indicate financial stability, while excessive mortgage obligations can increase repayment risk.",
        "example": "A borrower successfully managing one mortgage may appear financially stable compared to someone struggling with multiple mortgages."
    },

    "tot_cur_bal": {
        "display_name": "Total Current Balance",
        "what": "The total outstanding balance across all credit accounts.",
        "why": "It reflects the borrower's overall debt burden.",
        "effect_on_prediction": "Higher balances may indicate greater financial obligations, particularly when income is relatively low.",
        "example": "A borrower owing $80,000 across all accounts generally carries more debt than one owing $8,000."
    },

    "tot_hi_cred_lim": {
        "display_name": "Total Credit Limit",
        "what": "The combined credit limit across all accounts.",
        "why": "It represents the borrower's total available credit capacity.",
        "effect_on_prediction": "Higher credit limits can indicate stronger creditworthiness when utilization remains low.",
        "example": "Having $100,000 of available credit while using only a small portion is generally viewed positively."
    },

    "avg_cur_bal": {
        "display_name": "Average Account Balance",
        "what": "The average outstanding balance across all credit accounts.",
        "why": "It helps measure how heavily each account is utilized on average.",
        "effect_on_prediction": "Higher average balances may suggest greater financial commitments.",
        "example": "Average balances of $10,000 per account generally indicate heavier borrowing than averages of $500."
    },

    "acc_open_past_24mths": {
        "display_name": "Accounts Opened (Last 24 Months)",
        "what": "The number of new credit accounts opened during the past two years.",
        "why": "Opening many new accounts in a short period may indicate increasing borrowing activity.",
        "effect_on_prediction": "A large number of recently opened accounts may increase predicted risk.",
        "example": "Opening eight new accounts within two years generally reflects more aggressive credit usage than opening one."
    },

    # ===========================
    # Recent Credit Activity
    # ===========================

    "total_bal_ex_mort": {
        "display_name": "Total Balance Excluding Mortgage",
        "what": "The borrower's total outstanding balance excluding mortgage loans.",
        "why": "It measures non-housing debt obligations.",
        "effect_on_prediction": "Higher non-mortgage debt may increase repayment pressure.",
        "example": "Large personal loan and credit card balances can significantly increase this value."
    },

    "total_bc_limit": {
        "display_name": "Total Bankcard Credit Limit",
        "what": "The combined credit limit across all bankcard accounts.",
        "why": "It reflects the borrower's available revolving credit capacity.",
        "effect_on_prediction": "Higher limits can indicate stronger credit standing when balances remain controlled.",
        "example": "A borrower with $50,000 in available card limits has greater borrowing capacity than one with $5,000."
    },

    "bc_open_to_buy": {
        "display_name": "Available Bankcard Credit",
        "what": "The unused credit currently available on bankcard accounts.",
        "why": "Available credit indicates remaining borrowing capacity.",
        "effect_on_prediction": "More available credit generally reflects lower utilization and healthier credit management.",
        "example": "Having $20,000 of unused credit is generally stronger than having only $500 available."
    },

    "bc_util": {
        "display_name": "Bankcard Utilization",
        "what": "The percentage of available bankcard credit currently being used.",
        "why": "It measures how heavily credit cards are utilized.",
        "effect_on_prediction": "Higher utilization has historically been associated with increased default risk.",
        "example": "Using 95% of available card limits generally represents higher risk than using only 15%."
    },

    "num_actv_bc_tl": {
        "display_name": "Active Bankcard Accounts",
        "what": "The number of bankcard accounts that are currently active.",
        "why": "It reflects how many credit card accounts the borrower actively manages.",
        "effect_on_prediction": "Very high numbers of active bankcards may indicate greater financial complexity.",
        "example": "Managing three active cards is generally simpler than managing fifteen."
    },

    # ===========================
    # Tradeline Details
    # ===========================

    "num_bc_sats": {
        "display_name": "Satisfactory Bankcard Accounts",
        "what": "The number of bankcard accounts that are in good standing.",
        "why": "Well-managed credit accounts demonstrate responsible borrowing behavior.",
        "effect_on_prediction": "More satisfactory accounts generally strengthen the borrower's credit profile.",
        "example": "Ten well-managed credit cards provide stronger repayment evidence than only one."
    },

    "num_bc_tl": {
        "display_name": "Total Bankcard Accounts",
        "what": "The total number of bankcard accounts ever opened.",
        "why": "It reflects the borrower's experience with revolving credit.",
        "effect_on_prediction": "Its impact depends on whether the accounts have been managed responsibly.",
        "example": "Someone with many successfully managed cards may have a stronger credit history."
    },

    "num_il_tl": {
        "display_name": "Installment Accounts",
        "what": "The total number of installment loan accounts.",
        "why": "Installment accounts provide information about repayment experience on fixed-payment loans.",
        "effect_on_prediction": "A healthy installment history may strengthen the overall credit profile.",
        "example": "Successfully repaying multiple auto loans demonstrates installment repayment experience."
    },

    "num_op_rev_tl": {
        "display_name": "Open Revolving Accounts",
        "what": "The number of revolving credit accounts currently open.",
        "why": "It reflects the borrower's active revolving credit obligations.",
        "effect_on_prediction": "Too many open revolving accounts may indicate increased borrowing exposure.",
        "example": "Maintaining twenty active revolving accounts generally creates more complexity than maintaining five."
    },

    "num_rev_accts": {
        "display_name": "Total Revolving Accounts",
        "what": "The total number of revolving accounts ever opened.",
        "why": "It summarizes the borrower's revolving credit history.",
        "effect_on_prediction": "The effect depends on how responsibly those accounts have been managed.",
        "example": "A long history of well-managed revolving accounts may strengthen creditworthiness."
    },

    "num_actv_rev_tl": {
        "display_name": "Active Revolving Accounts",
        "what": "The number of revolving accounts currently active.",
        "why": "Active revolving accounts contribute to ongoing debt management responsibilities.",
        "effect_on_prediction": "A high number of active revolving accounts may increase repayment risk when combined with high balances.",
        "example": "Actively using many revolving accounts may increase financial obligations."
    },

    "num_tl_op_past_12m": {
        "display_name": "Accounts Opened (Last 12 Months)",
        "what": "The number of credit accounts opened during the previous year.",
        "why": "Rapid credit expansion may indicate increased borrowing activity.",
        "effect_on_prediction": "Opening many accounts recently may be associated with higher predicted risk.",
        "example": "Opening six new accounts within one year generally reflects more aggressive borrowing than opening one."
    },

    "percent_bc_gt_75": {
        "display_name": "Bankcards Above 75% Utilization",
        "what": "The percentage of bankcard accounts using more than 75% of their available credit.",
        "why": "Highly utilized cards may indicate financial pressure.",
        "effect_on_prediction": "Higher percentages generally correspond to increased default risk.",
        "example": "Using most available credit across multiple cards is typically riskier than maintaining low balances."
    },

    "pct_tl_nvr_dlq": {
        "display_name": "Accounts Never Delinquent",
        "what": "The percentage of credit accounts that have never experienced a delinquent payment.",
        "why": "It summarizes the borrower's long-term repayment consistency.",
        "effect_on_prediction": "Higher percentages generally indicate stronger repayment behavior.",
        "example": "A borrower with 100% clean payment history generally presents lower risk."
    },

    # ===========================
    # Severe Delinquency Indicators
    # ===========================

    "num_tl_90g_dpd_24m": {
        "display_name": "90+ Days Past Due (Last 24 Months)",
        "what": "The number of accounts that were at least 90 days overdue during the past two years.",
        "why": "Severe delinquencies strongly indicate repayment difficulties.",
        "effect_on_prediction": "Higher values are generally associated with significantly higher default risk.",
        "example": "Several accounts overdue by more than 90 days suggest substantial financial stress."
    },

    "num_tl_30dpd": {
        "display_name": "Current 30-Day Delinquencies",
        "what": "The number of accounts currently at least 30 days overdue.",
        "why": "Current payment delays indicate ongoing repayment problems.",
        "effect_on_prediction": "Higher numbers generally increase predicted default risk.",
        "example": "A borrower currently behind on multiple payments represents elevated credit risk."
    },

    "num_tl_120dpd_2m": {
        "display_name": "120+ Days Past Due (Last 2 Months)",
        "what": "The number of accounts that became at least 120 days overdue during the past two months.",
        "why": "This represents very serious recent repayment problems.",
        "effect_on_prediction": "Even a small number of such severe delinquencies can substantially increase predicted risk.",
        "example": "A loan overdue by four months indicates significant repayment difficulty."
    },

    "num_accts_ever_120_pd": {
        "display_name": "Accounts Ever 120+ Days Past Due",
        "what": "The number of accounts that have ever been at least 120 days overdue.",
        "why": "Historical severe delinquencies provide evidence of past financial distress.",
        "effect_on_prediction": "Higher counts generally correspond to greater predicted default risk.",
        "example": "Repeated severe delinquencies suggest previous repayment challenges."
    },

    "collections_12_mths_ex_med": {
        "display_name": "Collections (Excluding Medical)",
        "what": "The number of collection events during the last 12 months, excluding medical collections.",
        "why": "Recent collections indicate unresolved financial obligations.",
        "effect_on_prediction": "More collection events generally increase predicted default risk.",
        "example": "Recent collection activity may suggest ongoing repayment difficulties."
    },

    "chargeoff_within_12_mths": {
        "display_name": "Charge-Offs (Last 12 Months)",
        "what": "The number of accounts charged off during the previous year.",
        "why": "Charge-offs occur when lenders consider debt unlikely to be repaid.",
        "effect_on_prediction": "Recent charge-offs are among the strongest indicators of elevated credit risk.",
        "example": "A recent charge-off reflects a significant past repayment failure."
    },

        # ===========================
    # Account Age & Recency
    # ===========================

    "mo_sin_old_rev_tl_op": {
        "display_name": "Age of Oldest Revolving Account",
        "what": "The number of months since the borrower's oldest revolving credit account was opened.",
        "why": "Older revolving accounts provide a longer history of credit management.",
        "effect_on_prediction": "Long-established revolving credit histories are generally associated with lower default risk.",
        "example": "A borrower with a credit card opened 15 years ago typically has a longer repayment history than someone who opened one last year."
    },

    "mo_sin_old_il_acct": {
        "display_name": "Age of Oldest Installment Account",
        "what": "The number of months since the oldest installment loan account was opened.",
        "why": "A longer installment credit history gives more evidence of repayment behavior.",
        "effect_on_prediction": "Older installment accounts generally strengthen the borrower's credit profile.",
        "example": "Successfully managing installment loans over many years often reflects responsible borrowing."
    },

    "mo_sin_rcnt_rev_tl_op": {
        "display_name": "Months Since Recent Revolving Account",
        "what": "The number of months since the most recent revolving account was opened.",
        "why": "Recently opened revolving accounts may indicate increasing borrowing activity.",
        "effect_on_prediction": "Very recent revolving accounts may slightly increase predicted default risk.",
        "example": "Opening a new credit card last month may indicate more active credit usage than opening one three years ago."
    },

    "mo_sin_rcnt_tl": {
        "display_name": "Months Since Recent Credit Account",
        "what": "The number of months since any credit account was most recently opened.",
        "why": "It measures how recently the borrower expanded their credit portfolio.",
        "effect_on_prediction": "Smaller values may indicate recent borrowing activity and can slightly increase predicted risk.",
        "example": "Opening multiple accounts within the past few months reflects recent credit expansion."
    },

    "mths_since_recent_bc": {
        "display_name": "Months Since Recent Bankcard",
        "what": "The number of months since the borrower most recently opened a bankcard account.",
        "why": "Recent bankcard openings may indicate increasing reliance on revolving credit.",
        "effect_on_prediction": "Very recent bankcard activity may contribute modestly to higher predicted risk.",
        "example": "Opening a new credit card last month is generally more significant than opening one five years ago."
    },

    "mths_since_recent_bc_dlq": {
        "display_name": "Months Since Recent Bankcard Delinquency",
        "what": "The number of months since the borrower last became delinquent on a bankcard account.",
        "why": "Recent missed credit card payments are important indicators of repayment behavior.",
        "effect_on_prediction": "More recent delinquencies generally correspond to higher default risk.",
        "example": "A missed payment two months ago is usually more concerning than one from six years ago."
    },

    "mths_since_recent_inq": {
        "display_name": "Months Since Recent Credit Inquiry",
        "what": "The number of months since the borrower most recently applied for credit.",
        "why": "Frequent or recent credit applications may indicate increased borrowing needs.",
        "effect_on_prediction": "Very recent inquiries may modestly increase predicted risk when combined with other factors.",
        "example": "A borrower applying for several loans within a short period may appear more financially stressed."
    },

    "mths_since_recent_revol_delinq": {
        "display_name": "Months Since Recent Revolving Delinquency",
        "what": "The number of months since the borrower last became delinquent on a revolving account.",
        "why": "It measures how recently repayment issues occurred on revolving credit.",
        "effect_on_prediction": "Recent revolving delinquencies generally indicate elevated repayment risk.",
        "example": "A revolving delinquency from last year is generally more relevant than one from ten years ago."
    },

    "mths_since_last_major_derog": {
        "display_name": "Months Since Last Major Derogatory Event",
        "what": "The number of months since the borrower's last major negative credit event.",
        "why": "Major derogatory events are strong indicators of previous financial distress.",
        "effect_on_prediction": "More recent major derogatory events generally increase predicted default risk.",
        "example": "A serious credit event within the past year typically has greater impact than one many years ago."
    },

    # ===========================
    # Bureau Installment Features
    # ===========================

    "open_acc_6m": {
        "display_name": "Accounts Opened (Last 6 Months)",
        "what": "The number of credit accounts opened during the previous six months.",
        "why": "Rapid credit expansion may indicate increased borrowing activity.",
        "effect_on_prediction": "Opening several new accounts in a short period may modestly increase predicted risk.",
        "example": "Opening five accounts in six months generally represents more aggressive borrowing than opening one."
    },

    "open_act_il": {
        "display_name": "Open Installment Accounts",
        "what": "The number of installment loan accounts that are currently open.",
        "why": "It reflects the borrower's current installment loan obligations.",
        "effect_on_prediction": "The impact depends on repayment history and overall debt burden.",
        "example": "Managing multiple auto or personal loans successfully may demonstrate repayment experience."
    },

    "open_il_12m": {
        "display_name": "Installment Accounts Opened (Last 12 Months)",
        "what": "The number of installment accounts opened during the previous year.",
        "why": "Recent installment borrowing provides insight into current borrowing activity.",
        "effect_on_prediction": "Opening many installment loans recently may indicate increasing debt obligations.",
        "example": "Several new personal loans within one year may increase repayment pressure."
    },

    "open_il_24m": {
        "display_name": "Installment Accounts Opened (Last 24 Months)",
        "what": "The number of installment accounts opened during the previous two years.",
        "why": "It reflects medium-term borrowing activity.",
        "effect_on_prediction": "A higher number of recently opened installment loans may increase predicted risk.",
        "example": "Multiple auto and personal loans opened over two years may indicate expanding debt."
    },

    "mths_since_rcnt_il": {
        "display_name": "Months Since Recent Installment Loan",
        "what": "The number of months since the borrower most recently opened an installment account.",
        "why": "It measures how recently installment borrowing occurred.",
        "effect_on_prediction": "More recent installment borrowing may slightly increase repayment risk.",
        "example": "Opening an installment loan last month indicates more recent borrowing than opening one four years ago."
    },

    "total_bal_il": {
        "display_name": "Total Installment Balance",
        "what": "The total outstanding balance across all installment loans.",
        "why": "It reflects the borrower's existing installment debt burden.",
        "effect_on_prediction": "Higher installment balances may increase financial obligations, particularly when income is limited.",
        "example": "Large auto and personal loan balances increase the total installment balance."
    },

    "il_util": {
        "display_name": "Installment Utilization",
        "what": "The percentage of available installment credit currently being used.",
        "why": "Higher utilization may indicate heavier reliance on installment borrowing.",
        "effect_on_prediction": "Higher utilization levels are often associated with increased repayment risk.",
        "example": "Using most available installment credit leaves less financial flexibility."
    },

    "open_rv_12m": {
        "display_name": "Revolving Accounts Opened (Last 12 Months)",
        "what": "The number of revolving accounts opened during the previous year.",
        "why": "Recent revolving credit activity provides insight into borrowing behavior.",
        "effect_on_prediction": "Opening many revolving accounts recently may modestly increase predicted risk.",
        "example": "Opening several new credit cards within one year reflects active borrowing."
    },

    "open_rv_24m": {
        "display_name": "Revolving Accounts Opened (Last 24 Months)",
        "what": "The number of revolving accounts opened during the previous two years.",
        "why": "It measures medium-term revolving credit growth.",
        "effect_on_prediction": "A higher number of recently opened revolving accounts may indicate increased credit demand.",
        "example": "Opening numerous revolving accounts over two years may suggest expanding credit usage."
    },

    "max_bal_bc": {
        "display_name": "Maximum Bankcard Balance",
        "what": "The highest balance recorded on any bankcard account.",
        "why": "Large historical balances may indicate periods of heavy credit usage.",
        "effect_on_prediction": "Higher maximum balances may increase predicted risk when supported by other credit indicators.",
        "example": "A borrower who once carried a $15,000 card balance may have experienced higher borrowing pressure."
    },

    "all_util": {
        "display_name": "Overall Credit Utilization",
        "what": "The percentage of total available credit currently being used across all applicable accounts.",
        "why": "Overall utilization summarizes how heavily the borrower relies on available credit.",
        "effect_on_prediction": "Higher utilization is generally associated with increased default risk.",
        "example": "Using 90% of available credit is generally riskier than using only 20%."
    },

    "inq_fi": {
        "display_name": "Finance Company Inquiries",
        "what": "The number of recent credit inquiries made by finance companies.",
        "why": "Frequent finance company inquiries may indicate active borrowing attempts.",
        "effect_on_prediction": "Higher inquiry counts may modestly increase predicted default risk.",
        "example": "Multiple finance company inquiries within a short period may suggest increased credit demand."
    },

    "total_cu_tl": {
        "display_name": "Credit Union / Finance Accounts",
        "what": "The total number of credit union or finance-related accounts.",
        "why": "It contributes to the borrower's overall account mix.",
        "effect_on_prediction": "Its effect depends on how those accounts have been managed over time.",
        "example": "A healthy mix of responsibly managed account types can strengthen the overall credit profile."
    },

    "inq_last_12m": {
        "display_name": "Credit Inquiries (Last 12 Months)",
        "what": "The total number of credit inquiries made during the previous year.",
        "why": "Frequent credit applications may indicate increased borrowing needs.",
        "effect_on_prediction": "Higher inquiry counts are often associated with elevated default risk when combined with other indicators.",
        "example": "Ten inquiries in one year generally indicate more active credit seeking than one inquiry."
    },

    "total_il_high_credit_limit": {
        "display_name": "Total Installment Credit Limit",
        "what": "The combined credit limit across all installment loan accounts.",
        "why": "It reflects the borrower's available installment credit capacity.",
        "effect_on_prediction": "Higher installment credit limits may be positive when balances remain well managed and repayment history is strong.",
        "example": "A borrower with substantial available installment credit and low outstanding balances generally demonstrates stronger financial capacity."
    }
}
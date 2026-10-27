"""
utils/canonical_map.py
──────────────────────
Single source of truth for US-GAAP tag → canonical field normalization.

This mapping is used by:
- ingestion/xbrl_parser.py (local XBRL file parsing)
- ingestion/sec_fetcher.py (SEC EDGAR API JSON facts)

Keep in sync — do not duplicate in other files.
"""

# Raw US-GAAP tag → normalized canonical field
CANONICAL_MAP: dict[str, str] = {
    # Revenue & Sales
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",

    # Cost of Sales & Gross Profit
    "GrossProfit": "gross_profit",
    "CostOfRevenue": "cost_of_revenue",
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "CostOfGoodsSold": "cost_of_revenue",

    # Operating Income & Expenses
    "OperatingIncomeLoss": "operating_income",
    "OperatingExpenses": "operating_expenses",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    "SellingAndMarketingExpense": "marketing_expense",
    "GeneralAndAdministrativeExpense": "ga_expense",

    # Pre-tax, Taxes & Net Income
    "NetIncomeLoss": "net_income",
    "ProfitLoss": "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "pretax_income",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "IncomeTaxExpenseBenefit": "income_tax_expense",
    "InterestExpense": "interest_expense",
    "InterestAndDebtExpense": "interest_expense",
    "InterestIncomeExpenseNet": "interest_income",
    "InvestmentIncomeNonoperating": "interest_income",
    "DepreciationDepletionAndAmortization": "depreciation_amortization",
    "DepreciationAndAmortization": "depreciation_amortization",
    "Depreciation": "depreciation_amortization",
    "AmortizationOfIntangibleAssets": "depreciation_amortization",

    # EPS & Shares
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic": "shares_outstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "shares_outstanding",

    # Balance Sheet — Assets
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashAndCashEquivalentsAtCarryingValueIncludingCashEquivalents": "cash",
    "CashCashEquivalentsAndShortTermInvestments": "cash_and_investments",
    "ShortTermInvestments": "short_term_investments",
    "MarketableSecuritiesCurrent": "short_term_investments",
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "InventoryNet": "inventory",
    "OtherAssetsCurrent": "other_current_assets",
    "PropertyPlantAndEquipmentNet": "ppe_net",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "DeferredTaxAssetsNet": "deferred_tax_assets",
    "OtherAssetsNoncurrent": "other_noncurrent_assets",

    # Balance Sheet — Liabilities & Equity
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    "AccountsPayableCurrent": "accounts_payable",
    "AccruedLiabilitiesCurrent": "accrued_liabilities",
    "DeferredRevenueCurrent": "deferred_revenue_current",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "DeferredTaxLiabilitiesNet": "deferred_tax_liabilities",
    "OtherLiabilitiesNoncurrent": "other_noncurrent_liabilities",
    "StockholdersEquity": "stockholders_equity",
    "CommonStockValue": "common_stock",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax": "accumulated_oci",

    # Cash Flow
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
    "NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsForRepurchaseOfCommonStock": "share_repurchases",
    "ProceedsFromIssuanceOfCommonStock": "stock_issuance",
    "PaymentsOfDividendsCommonStock": "dividends_paid",
    "PaymentsOfDividends": "dividends_paid",
    "ProceedsFromLongTermDebt": "debt_issuance",
    "RepaymentsOfLongTermDebt": "debt_repayment",

    # Additional Income Statement
    "OtherIncomeExpenseNet": "other_income_net",
    "IncomeLossFromDiscontinuedOperationsNetOfTax": "discontinued_operations",

    # Per-share / ratios (computed, not directly tagged but useful)
    "BookValuePerShare": "book_value_per_share",
}

# Core investor-facing facts to surface in RAG prompts
DEFAULT_CANONICAL_FIELDS = [
    "revenue",
    "gross_profit",
    "cost_of_revenue",
    "operating_income",
    "operating_expenses",
    "rd_expense",
    "sga_expense",
    "pretax_income",
    "income_tax_expense",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "shares_outstanding",
    "total_assets",
    "current_assets",
    "cash",
    "cash_and_investments",
    "total_liabilities",
    "current_liabilities",
    "long_term_debt",
    "stockholders_equity",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
    "share_repurchases",
    "dividends_paid",
]

__all__ = ["CANONICAL_MAP", "DEFAULT_CANONICAL_FIELDS"]
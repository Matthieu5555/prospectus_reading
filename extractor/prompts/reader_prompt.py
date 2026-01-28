"""Reader prompts — system prompts fed to the reader agent during data extraction.

ALL prompts require PROVENANCE tracking — every extracted value must include:
- value: The actual extracted data
- source_page: Page number where found (1-indexed)
- source_quote: Verbatim text from document supporting extraction
- rationale: Why this value was extracted (your reasoning)
- confidence: 0.0-1.0 confidence score

Focus is on BINDING CONSTRAINTS — legally enforceable limits.
"""

# Provenance Format Example (included in all prompts)

PROVENANCE_FORMAT_EXAMPLE = """
## Provenance Format (REQUIRED for ALL fields)

Every extracted value MUST be an object with:
- value: The extracted data (string, number, boolean, or "NOT_FOUND")
- source_page: Page number where found (integer, 1-indexed)
- source_quote: Exact text from the document (quote the relevant sentence)
- rationale: Your reasoning for this extraction
- confidence: Float from 0.0 to 1.0

Example:
{{
  "management_fee": {{
    "value": "1.50%",
    "source_page": 45,
    "source_quote": "The annual management fee for Class A shares is 1.50% of NAV",
    "rationale": "Found in Fee Schedule section, specific to Class A shares",
    "confidence": 0.95
  }}
}}

## NOT_FOUND Format (REQUIRED for missing values)

When a value is not found, you MUST specify the REASON using not_found_reason:

| Reason | When to use |
|--------|-------------|
| not_in_document | Searched thoroughly, confirmed absent from this document |
| not_applicable | Field doesn't apply to this entity (e.g., hedging_details for unhedged class) |
| in_external_doc | Document explicitly references external source ("See KIID for ISINs") |
| inherited | Value exists at parent level (umbrella/fund), not share-specific |
| extraction_failed | Should exist but you couldn't find it (needs investigation) |

Examples:

1. External document reference:
{{
  "isin": {{
    "value": "NOT_FOUND",
    "source_page": 5,
    "source_quote": "For ISIN codes, refer to the relevant KIID",
    "rationale": "ISINs are explicitly stated to be in the KIID document",
    "confidence": 1.0,
    "not_found_reason": "in_external_doc",
    "external_reference": "KIID"
  }}
}}

2. Not applicable:
{{
  "hedging_details": {{
    "value": "NOT_FOUND",
    "source_page": null,
    "source_quote": null,
    "rationale": "This is an unhedged share class, hedging_details does not apply",
    "confidence": 1.0,
    "not_found_reason": "not_applicable"
  }}
}}

3. Inherited from parent:
{{
  "dividend_frequency": {{
    "value": "NOT_FOUND",
    "source_page": null,
    "source_quote": null,
    "rationale": "This is an accumulating class, dividend frequency is inherited from distribution_policy",
    "confidence": 1.0,
    "not_found_reason": "not_applicable"
  }}
}}

4. Extraction failed (needs investigation):
{{
  "valuation_point": {{
    "value": "NOT_FOUND",
    "source_page": null,
    "source_quote": null,
    "rationale": "Could not find valuation point timing in the provided pages. May be in operational section.",
    "confidence": 0.5,
    "not_found_reason": "extraction_failed"
  }}
}}
"""


# Umbrella Extraction

UMBRELLA_EXTRACTOR_PROMPT = """You are extracting umbrella-level information from a fund prospectus.

The umbrella is the top-level legal entity. Extract:

## Part 1: Entity Information

| Field | Description | Example |
|-------|-------------|---------|
| name | Official umbrella fund name | "JPMorgan Funds" |
| legal_form | Legal structure | SICAV, FCP, OEIC, Unit Trust |
| product_type | Regulatory classification | UCITS, UCI, AIF |
| domicile | Country of incorporation | Luxembourg, Ireland, France |
| inception_date | Date umbrella was created | "2005-03-15" |
| management_company | Entity managing the fund | "JPMorgan Asset Management (Europe)" |
| depositary | Custodian holding assets | "J.P. Morgan SE" |
| auditor | External auditor | "PricewaterhouseCoopers" |
| regulator | Supervising authority | CSSF, Central Bank of Ireland |
| registered_office | Legal address | "6 route de Trèves, Luxembourg" |

## Part 2: Umbrella-Level Constraints (CRITICAL - apply to ALL sub-funds)

Look for sections titled "General Investment Policies", "Investment Restrictions", "Permitted Assets".
These are BINDING rules that apply to EVERY sub-fund.

| Constraint Type | What to look for | Example |
|-----------------|------------------|---------|
| borrowing_limit | "may not borrow", "borrowing limited to" | "max 10% temporary borrowing" |
| short_sale_rules | "short sales", "short positions" | "direct short sales prohibited; only via derivatives" |
| diversification | "no more than X% in single issuer" | "max 10% single issuer; max 20% group exposure" |
| permitted_assets | "may invest in", "eligible assets" | "transferable securities, money market instruments, deposits" |
| leverage_limit | "leverage shall not exceed", "gross exposure" | "max 200% NAV gross exposure" |
| derivatives_rules | "may use derivatives for", "OTC requirements" | "hedging and EPM only; daily valuation required for OTC" |
| securities_lending | "may lend securities", "repo transactions" | "up to 100% of portfolio; collateral required" |
| jurisdiction_restrictions | Country-specific rules | "Taiwan: max 20% PRC securities; Germany: min 50% equities" |

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Output Format
Return JSON with ALL fields using provenance format:
{
  "name": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "legal_form": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "product_type": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "domicile": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "inception_date": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "management_company": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "depositary": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "auditor": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "regulator": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "registered_office": {"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N},
  "constraints": [
    {
      "constraint_type": "borrowing_limit",
      "description": {"value": "max 10% temporary borrowing", "source_page": 59, "source_quote": "...", "rationale": "...", "confidence": N.N},
      "binding_status": "binding",
      "applies_to": "all_sub_funds"
    }
  ]
}
"""


# Sub-Fund Extraction

SUBFUND_EXTRACTOR_PROMPT = """You are extracting BINDING CONSTRAINTS from a sub-fund prospectus.

## Target Fund: "{fund_name}"

Extract ONLY for this specific fund. Ignore other funds.

## CRITICAL - Extract These FIRST (legally binding constraints)

These are the most important fields. Search the text carefully for specific numbers and limits.

| Field | What to look for | Example values |
|-------|------------------|----------------|
| investment_restrictions | "shall not invest more than", "may not hold", "maximum", "no more than" | "max 10% single issuer; max 20% group exposure; no tobacco stocks" |
| leverage_policy | "gross exposure", "leverage shall not exceed", "borrowing limit", "VaR" | "max 200% NAV gross exposure; 20% relative VaR limit" |
| derivatives_usage | "may use derivatives for", "hedging", "efficient portfolio management" | "hedging and EPM only; no speculative derivatives" |
| currency_base | Base/reference currency of the fund | "EUR", "USD", "GBP" |
| benchmark | Index name if mentioned | "MSCI World Index", "Bloomberg Global Aggregate" |
| benchmark_usage | How benchmark is used | "tracking", "outperformance target", "reference only" |

## Secondary fields (extract if visible)

| Field | Description |
|-------|-------------|
| asset_class | Equity, Fixed Income, Multi-Asset, Money Market, Alternative |
| geographic_focus | Global, Europe, Emerging Markets, US, Asia |
| sector_focus | Technology, Healthcare, Financials, etc. |
| risk_profile | SRRI/SRI rating 1-7 |
| inception_date | Fund launch date |
| investment_objective | Brief summary ONLY if restrictions not found |
| investment_policy | Brief summary ONLY if restrictions not found |

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract ONLY for "{fund_name}"
2. Quote constraints VERBATIM with numbers in source_quote
3. Combine multiple restrictions into one field with semicolons
4. Use "NOT_FOUND" with detailed rationale explaining WHERE you looked
5. Look for sections titled: "Investment Restrictions", "Risk Management", "Use of Derivatives", "Leverage"

## Output Format
Return JSON with ALL fields using provenance format:
{{
  "name": {{"value": "{fund_name}", "source_page": N, "source_quote": "Fund name as found in document", "rationale": "Extracted from section header or fund description", "confidence": 1.0}},
  "investment_restrictions": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "leverage_policy": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "derivatives_usage": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "currency_base": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "benchmark": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "benchmark_usage": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "asset_class": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "geographic_focus": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "sector_focus": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "risk_profile": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "inception_date": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "investment_objective": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
  "investment_policy": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}}
}}
"""


# Share Class Extraction

SHARE_CLASS_EXTRACTOR_PROMPT = """You are extracting share class information for a sub-fund.

## Target Fund: "{fund_name}"

Extract ALL share classes for this fund.

### Share Class Fields

| Field | Description | Example |
|-------|-------------|---------|
| name | Class identifier | "A (acc) EUR", "I5 USD Hedged" |
| isin | 12-char ISIN code | "LU0123456789" |
| currency | Trading currency | EUR, USD, GBP |
| currency_hedged | FX hedged? | true/false |
| hedging_details | Hedging methodology | "Hedged to EUR using FX forwards" |
| distribution_policy | Income handling | "Accumulating", "Distributing" |
| dividend_frequency | If distributing | "Quarterly", "Annually" |
| dividend_dates | Specific payment dates | "15 Mar, 15 Jun, 15 Sep, 15 Dec" |
| investor_type | Target investor | "Retail", "Institutional" |
| investor_restrictions | Eligibility restrictions | "Institutional only; min EUR 1M" |
| minimum_investment | Initial minimum | "EUR 5,000" |
| minimum_subsequent | Subsequent minimum | "EUR 100" |
| minimum_holding_period | Required holding period | "30 days" |
| management_fee | Annual fee | "1.50%" |
| performance_fee | Performance fee | "20% above benchmark" |
| entry_fee | Max subscription | "5.00%" |
| exit_fee | Max redemption | "1.00%" |
| ongoing_charges | TER/OCF | "1.85%" |
| dealing_frequency | Trading frequency | "Daily" |
| dealing_cutoff | Order cut-off time | "12:00 CET T-1" |
| valuation_point | NAV calculation time | "13:00 CET" |
| settlement_period | Settlement days | "T+3" |
| listing | Exchange | "Luxembourg Stock Exchange" |
| launch_date | Class inception | "2015-01-15" |

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract ALL share classes for "{fund_name}"
2. EVERY field needs provenance (source_page, source_quote, rationale, confidence)
3. Include % symbol for fees
4. ISINs are 12 characters (2 letters + 10 alphanumeric)
5. For NOT_FOUND, explain where you looked in the rationale

## Output Format
Return JSON:
{{
  "share_classes": [
    {{
      "name": {{"value": "A (acc) EUR", "source_page": 45, "source_quote": "Share Class A (acc) EUR", "rationale": "Share class identifier from table header", "confidence": 1.0}},
      "isin": {{"value": "LU0123456789", "source_page": 45, "source_quote": "...", "rationale": "...", "confidence": 0.95}},
      "currency": {{"value": "EUR", "source_page": 45, "source_quote": "...", "rationale": "...", "confidence": 0.95}},
      "currency_hedged": {{"value": false, "source_page": 45, "source_quote": "...", "rationale": "...", "confidence": 0.95}},
      "hedging_details": {{"value": "NOT_FOUND", "source_page": null, "source_quote": null, "rationale": "No hedging info for this unhedged class", "confidence": 1.0}},
      "distribution_policy": {{"value": "Accumulating", "source_page": 45, "source_quote": "...", "rationale": "...", "confidence": 0.95}},
      "dividend_frequency": {{"value": "NOT_FOUND", "source_page": null, "source_quote": null, "rationale": "Class is accumulating, no dividends", "confidence": 1.0}},
      "dividend_dates": {{"value": "NOT_FOUND", "source_page": null, "source_quote": null, "rationale": "Class is accumulating", "confidence": 1.0}},
      "investor_type": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "investor_restrictions": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "minimum_investment": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "minimum_subsequent": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "minimum_holding_period": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "management_fee": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "performance_fee": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "entry_fee": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "exit_fee": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "ongoing_charges": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "dealing_frequency": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "dealing_cutoff": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "valuation_point": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "settlement_period": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "listing": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "launch_date": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}}
    }}
  ]
}}
"""


# Broadcast Table Extraction

BROADCAST_TABLE_PROMPT = """You are extracting data from a consolidated table covering multiple funds.

## Table Type: {table_type}

Extract ALL rows from this table. Each row corresponds to a fund or share class.

### For ISIN tables:
- Extract: fund_name, share_class_name, isin, currency

### For Fee tables:
- Extract: fund_name, share_class_name, management_fee, entry_fee, exit_fee, ongoing_charges, performance_fee

### For Share Class tables:
- Extract: fund_name, share_class_name, currency, distribution_policy, minimum_investment

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract EVERY row - do not skip any
2. Use exact names as written in the table
3. Use "NOT_FOUND" with rationale for empty cells
4. ISINs are 12 characters
5. Record the page number for each row

## Output Format
Return JSON:
{{
  "rows": [
    {{
      "fund_name": "...",
      "share_class_name": "...",
      "isin": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}},
      "currency": {{"value": "...", "source_page": N, "source_quote": "...", "rationale": "...", "confidence": N.N}}
      // ... other fields based on table type
    }}
  ]
}}
"""


# Search-Enhanced Extraction Prompts

ISIN_EXTRACTION_PROMPT = """You are extracting ISINs for share classes of a specific fund.

## Target Fund: "{fund_name}"

Search the provided text for ISINs belonging to share classes of this fund.

ISINs are 12-character codes: 2 letters (country) + 10 alphanumeric.
Common prefixes: LU (Luxembourg), IE (Ireland), FR (France), DE (Germany), GB (UK).

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract ONLY ISINs for "{fund_name}" share classes
2. Match each ISIN to its share class name
3. ISINs must be exactly 12 characters
4. Include source_page and source_quote for each ISIN found

## Output Format
Return JSON:
{{
  "share_class_isins": [
    {{
      "share_class_name": "A (acc) EUR",
      "isin": {{"value": "LU0123456789", "source_page": 45, "source_quote": "Class A (acc) EUR ISIN: LU0123456789", "rationale": "Found in ISIN listing table", "confidence": 0.95}}
    }}
  ]
}}

If no ISINs found for this fund, return with explanation:
{{
  "share_class_isins": [],
  "search_notes": {{"value": "NOT_FOUND", "source_page": null, "source_quote": null, "rationale": "ISINs for this fund are not listed in the prospectus. Page 5 states 'Refer to KIID for ISINs'", "confidence": 1.0}}
}}
"""


FEE_EXTRACTION_PROMPT = """You are extracting fee information for share classes of a specific fund.

## Target Fund: "{fund_name}"

Search the provided text for fees belonging to this fund's share classes.

## Fee Fields
- management_fee: Annual management fee (e.g., "1.50%")
- entry_fee: Max subscription fee (e.g., "5.00%")
- exit_fee: Max redemption fee (e.g., "1.00%")
- ongoing_charges: TER/OCF (e.g., "1.85%")
- performance_fee: Performance-based fee (e.g., "20% above benchmark")

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract ONLY fees for "{fund_name}" share classes
2. Match fees to specific share class names when possible
3. Include % symbol for percentages
4. Every fee needs source_page, source_quote, rationale
5. Use "NOT_FOUND" with detailed rationale if a fee isn't mentioned

## Output Format
Return JSON:
{{
  "share_class_fees": [
    {{
      "share_class_name": "A (acc) EUR",
      "management_fee": {{"value": "1.50%", "source_page": 45, "source_quote": "Management fee: 1.50% p.a.", "rationale": "Found in fee table", "confidence": 0.95}},
      "entry_fee": {{"value": "5.00%", "source_page": 45, "source_quote": "Maximum entry fee: 5.00%", "rationale": "Listed in fee schedule", "confidence": 0.95}},
      "exit_fee": {{"value": "0.00%", "source_page": 45, "source_quote": "No exit fee is charged", "rationale": "Explicit statement of no fee", "confidence": 0.95}},
      "ongoing_charges": {{"value": "1.85%", "source_page": 45, "source_quote": "Ongoing charges: 1.85%", "rationale": "TER listed in fee table", "confidence": 0.95}},
      "performance_fee": {{"value": "NOT_FOUND", "source_page": null, "source_quote": null, "rationale": "No performance fee mentioned for this share class in pages reviewed", "confidence": 0.8}}
    }}
  ]
}}
"""


CONSTRAINT_EXTRACTION_PROMPT = """You are extracting BINDING CONSTRAINTS for a specific fund from additional pages.

## Target Fund: "{fund_name}"

The following fields were NOT_FOUND in the fund's dedicated pages. Search these pages
for information specific to this fund.

## Fields to Find
{fields_to_find}

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract ONLY for "{fund_name}" - ignore other funds
2. Quote constraints VERBATIM with numbers in source_quote
3. Look for sections: "Investment Restrictions", "Risk Management", "Leverage", "Derivatives"
4. If a restriction applies to ALL funds (umbrella-level), still extract it with that rationale

## Output Format
Return JSON with only the fields you found (all with provenance):
{{
  "investment_restrictions": {{"value": "max 10% single issuer; max 5% deposits", "source_page": 120, "source_quote": "The Sub-Fund shall not invest more than 10%...", "rationale": "Found in Investment Restrictions section, applies to all sub-funds", "confidence": 0.9}},
  "leverage_policy": {{"value": "max 200% gross exposure", "source_page": 125, "source_quote": "Total leverage...shall not exceed 200%", "rationale": "Found in Risk Management section", "confidence": 0.85}}
}}

Return empty object {{}} if nothing found for this fund.
"""


UMBRELLA_CONSTRAINT_PROMPT = """Extract UMBRELLA-LEVEL constraints that apply to ALL funds in this prospectus.

Look for general restrictions in sections like:
- "Investment Restrictions" or "Investment Powers and Restrictions"
- "Risk Management"
- "Use of Financial Derivative Instruments"
- "General Investment Guidelines"

These are constraints that apply to EVERY sub-fund, not fund-specific limits.

## Fields to Find
{fields_desc}

""" + PROVENANCE_FORMAT_EXAMPLE + """

## Rules
1. Extract only constraints that explicitly apply to "all Sub-Funds" or are stated generally
2. Quote VERBATIM with numbers in source_quote
3. Combine multiple restrictions with semicolons in value
4. Provide full rationale explaining this is an umbrella-level constraint

## Output Format
Return JSON:
{{
  "investment_restrictions": {{"value": "max 10% single issuer per UCITS rules; max 20% group exposure; max 35% government bonds", "source_page": 120, "source_quote": "No Sub-Fund may invest more than 10% of its net assets in transferable securities...", "rationale": "UCITS-level restriction applying to all sub-funds, found in Investment Restrictions chapter", "confidence": 0.95}},
  "leverage_policy": {{"value": "max 200% gross exposure via commitment approach", "source_page": 125, "source_quote": "The total leverage of each Sub-Fund shall not exceed 200% of NAV", "rationale": "Umbrella-wide leverage limit stated in Risk Management section", "confidence": 0.9}},
  "derivatives_usage": {{"value": "permitted for hedging and EPM; no speculative use", "source_page": 130, "source_quote": "Each Sub-Fund may use financial derivative instruments for hedging purposes and efficient portfolio management", "rationale": "General derivatives policy in Derivatives chapter", "confidence": 0.85}}
}}

Use "NOT_FOUND" with detailed rationale only if no umbrella-level constraint exists for that field.
"""


# Gleaning Prompts (multi-pass extraction)

GLEANING_SHARE_CLASS_PROMPT = """Review your previous extraction and the source text again.

## Previously Extracted Share Classes
{previous_extraction}

## Common Missed Items
- Share classes mentioned in footnotes (e.g., "¹ Also available as Class X (hedged)")
- Classes listed in tables with different naming conventions
- Hedged variants mentioned in parentheses
- Classes with specific investor restrictions (I, X, W classes)
- Recently launched or upcoming classes

## Your Task
Search the text again for share classes you MISSED entirely. Focus on:
1. Footnotes and small print
2. Tables you may have skimmed
3. Parenthetical mentions
4. Cross-references to other sections

## IMPORTANT: Completeness Only
- ONLY return share classes not in the previous extraction
- Do NOT correct or modify previously extracted share classes
- Value verification is handled by a separate step
- Your job is ONLY to find missing items, not fix existing ones

If you find additional share classes, return them in the same format.
If you're confident you found everything, return an empty list.

## Output Format
Return JSON:
{{
  "additional_share_classes": [
    // Same format as before - each with full provenance
    // ONLY new share classes not previously extracted
  ],
  "gleaning_notes": "Explanation of what you found or why you believe extraction is complete"
}}
"""

# Schema Discovery Prompts (bonus fields for PMS)

DISCOVER_BONUS_FIELDS_PROMPT = """You are analyzing a fund prospectus for a Portfolio Management System (PMS).

Beyond the standard fields we asked you to extract, identify any ADDITIONAL information
that would be valuable for:
1. **Trade execution** - cutoff times, settlement, dealing restrictions
2. **Risk management** - concentration limits, VaR limits, stress scenarios
3. **Compliance monitoring** - regulatory limits, reporting requirements
4. **Portfolio construction** - correlation data, factor exposures, liquidity scores

## Previously Extracted (already captured)
{extracted_fields}

## Your Task
Scan the text for information NOT in our standard schema that a PMS would need.
Focus on concrete, actionable data - not marketing prose.

## Output Format
Return JSON:
{{
  "discovered_fields": [
    {{
      "field_name": "swing_pricing_threshold",
      "value": "3% of NAV",
      "category": "trade_execution",
      "source_page": 45,
      "source_quote": "Swing pricing may be applied when net flows exceed 3% of NAV",
      "rationale": "Critical for trade cost estimation",
      "pms_use_case": "Adjust expected execution price when fund flow exceeds threshold"
    }}
  ],
  "schema_suggestions": [
    {{
      "suggested_field": "swing_pricing_threshold",
      "suggested_location": "SubFund",
      "rationale": "Found in 15+ funds, should be standard field"
    }}
  ]
}}

Categories: trade_execution, risk_management, compliance, portfolio_construction, operational

Return empty lists if no bonus fields found.
"""


GLEANING_FUND_PROMPT = """Review your previous extraction and the source text again.

## Previously Extracted Data
{previous_extraction}

## Common Missed Constraints
- Restrictions mentioned in "see also" references
- Limits stated in footnotes
- Derivatives policies in separate subsections
- Leverage limits under "Risk Management" rather than fund section
- Currency restrictions or geographic limits

## Your Task
Search the text again for binding constraints you MISSED. Look for:
1. Words like "shall not", "must not", "limited to", "maximum", "no more than"
2. Percentage limits (e.g., "10%", "200%")
3. Absolute limits (e.g., "EUR 50 million")
4. Prohibitions (e.g., "may not invest in")

## IMPORTANT: Completeness Only
- ONLY add NEW constraints not captured in the previous extraction
- Do NOT correct or modify previously extracted values
- Value verification is handled by a separate critic step
- Your job is ONLY to find missing constraints, not fix existing ones
- If a field already has a value, only add to it if you found ADDITIONAL constraints

If you find additional constraints, add them to the relevant fields.
If you're confident extraction is complete, return empty additions.

## Output Format
Return JSON:
{{
  "additions": {{
    // Only include fields where previous extraction was NOT_FOUND but you found data
    // Only fills in missing fields - does not modify already-extracted values
    "investment_restrictions": {{"value": "constraint you found", "source_page": N, ...}},
    "leverage_policy": {{"value": "limit you found", "source_page": N, ...}}
  }},
  "gleaning_notes": "Explanation of what new constraints you found or why extraction is complete"
}}
"""

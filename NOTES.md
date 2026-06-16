# Data quirks and cleaning decisions

This is the reasoning behind the cleaning step. Each rule exists because something in
the raw data made it necessary, and this note links the two. The evidence comes from two
diagnostic scripts that were run once to understand the data before any rule was written:

- `exploration/profile_data.py` → `data_profile_report.txt`
- `exploration/investigate_issues.py` → `issue_investigation_report.txt`

Nothing here was assumed from the dataset's reputation. Every figure below is from those
two reports.

## What the raw data looked like

1,067,371 rows, eight columns, spanning 2009-12-01 to 2011-12-09. United Kingdom is
91.9% of rows; 43 country labels in total, including a few non-countries (`Unspecified`,
`European Community`). Two columns had missing values: `Description` (4,382 rows, 0.41%)
and `Customer ID` (243,007 rows, 22.77%).

## The findings, and what each one led to

### Dates were stored in ISO format, not day-first

The first profiling pass reported every one of the 1,067,371 dates as failing to parse.
The cause was a wrong assumption in the code, not bad data: the file stores dates as
`YYYY-MM-DD HH:MM:SS`, but the parser was told to expect `dd/mm/yyyy`. Corrected, all
rows parse and the range is 2009-12-01 to 2011-12-09. No data lost. Worth recording
because it's a reminder to check the format rather than trust it.

### 34,335 exact duplicate rows → drop them

3.22% of rows were complete duplicates. The investigation showed groups like the same
invoice, same product, same minute, same price, repeated. Quantity already records
multiples (a purchase of three is one row with quantity 3), so identical rows are
double-logging, not separate sales. **Rule:** drop exact duplicates, keep one. Done first,
before anything sums revenue.

### Cancellations and adjustments sit under invoice prefixes → drop by prefix and by sign

Invoice numbers carry meaning. A plain number is a sale; `C` is a customer cancellation;
`A` is an accounting adjustment. The investigation found:

- 19,494 cancellation rows (`C`), almost all with negative quantity (19,493 of them).
- 3,457 negative-quantity rows that are *not* cancellations — stock write-offs with
  descriptions like "lost", "damages", "short", at price 0.
- 5 negative-price rows, all "Adjust bad debt" on invoices starting `A`, including a
  −£53,594 entry.
- 6,202 zero-price rows (samples, freebies, write-offs).

**Rule:** keep only genuine sales — drop invoices starting `C` or `A`, and require
quantity > 0 and price > 0. The two filters together catch both the prefixed non-sales
and the write-offs that slipped in under ordinary invoice numbers.

### Non-product codes carry real prices → filter on the code pattern

Postage, manual adjustments, bank charges, Amazon fees, charity donations, gift
vouchers, and sample codes (`POST`, `M`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, `GIFT`,
`DCGS...`) aren't products a customer chose. Postage in particular has a positive price,
so the price filter won't remove it. Real product codes look like five digits with an
optional letter suffix (`85123A`, `10124C`). **Rule:** keep only codes matching that
pattern; the run prints the codes it drops so they can be eyeballed before committing.

### One stock code, several descriptions → one canonical name per code

24.9% of stock codes carried more than one description. Most cases were one real name
used thousands of times plus a few one-off junk entries ("FOUND", "DAMAGED", "?",
"AMAZON") — which are themselves adjustment rows that the sales filter removes. A real
case: code 21181 had "PLEASE ONE PERSON METAL SIGN" and the same with a double space, so
whitespace alone split one product into two. **Rule:** collapse internal whitespace, then
map every code to its most frequent description, so one product has one name. This
matters for basket analysis, where a product appearing under two names would break the
co-purchase counts.

### 4,382 missing descriptions → backfill where possible

Of these, 4,021 had a stock code that appears elsewhere *with* a description, so they can
be backfilled from the canonical name. The remaining 361 have product-looking codes that
never carry a description anywhere — 0.03% of the data, filled with a placeholder rather
than dropped.

### 22.77% missing customer IDs → drop for customer work, keep for basket

This was the biggest branch, and the investigation made the decision clear. Of the
243,007 rows with no customer ID, only 750 are cancellations; **236,122 look like
ordinary sales**, spread across 8,752 invoices. So this isn't junk — it's about a fifth
of real trade that simply can't be tied to a person.

**Rule:** drop these rows from RFM, clustering, and segmentation, which all need to
attribute behaviour to a customer. **Keep** them for basket analysis, where the unit is
the invoice and an anonymous basket is still a valid basket. Anonymous rows are 22.6% of
rows but only 13.1% of revenue, so they skew towards smaller transactions.

### Two phantom giant orders → removed before RFM

Two invoices stood out: 80,995 units of "PAPER CRAFT, LITTLE BIRDIE" (invoice 581483,
~£168k) and 74,215 units of a ceramic storage jar (invoice 541431, ~£77k). Both were
placed and cancelled the same day. Because cancellations were removed in cleaning, the
positive side survived and made two customers look like the biggest buyers in the data
when their real net was near zero. **Rule:** drop these two invoices before computing RFM.
They were identified and confirmed by inspecting the largest orders, not removed blindly.
The genuine wholesalers below them (a £581k and a £527k customer, buying repeatedly
across two years) are real and were kept.

## A choice worth stating: gross vs net monetary

Because cancellations were removed rather than matched to their original sales and
subtracted, the monetary value is **gross** — a customer who bought £10k and returned £4k
shows as £10k. This was a deliberate trade-off: gross is simple and easy to explain,
whereas netting returns means fuzzy matching rules that introduce their own errors. The
two phantom orders were the only cases extreme enough to distort the analysis, and they
were handled directly. The bias is documented rather than hidden, and netting remains an
option if the segmentation ever looks distorted by it.

## Why this matters

Most of these decisions are invisible in the final result, but each one shapes it. The
cleaning step removed cancellations, adjustments, non-products, duplicates, and two
phantom orders, and resolved the missing-ID question by splitting the data into two views
— one for customer work, one for basket work. The point of this note is that none of
those choices was a default: each came from a specific thing found in the data, and each
is defensible on its own terms.

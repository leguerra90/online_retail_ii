"""
Project: Customer Intelligence — pipeline entry point.

Run this to execute the project end to end:

    python main.py

Each step lives in its own file and follows the same shape: construct the
class with its input, call .run() (which returns self), read the result
property, call .save(). That keeps this file short and makes adding a step
a matter of uncommenting a block.

Steps:
    1. clean    retail_cleaner.py        RetailCleaner       
    2. rfm      retail_rfm.py            RFMBuilder          
    3. segment  retail_segmentation.py   CustomerSegmenter   
    4. basket   retail_basket.py         BasketAnalyser      

Data hand-offs:
    cleaner.customer_sales  ->  RFM and segmentation (need a Customer ID)
    cleaner.sales           ->  basket analysis (all valid invoices)
"""

from pathlib import Path

from retail_cleaner import RetailCleaner
from retail_eda import RetailEDA
from retail_rfm import RFMBuilder
from retail_segmentation import CustomerSegmenter
from retail_basket import BasketAnalyser    

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = Path("kaggle_customer_intelligence")
RAW_FILE = DATA_DIR / "online_retail_II.csv"


def main():
    # ------------------------------------------------------------------
    # STEP 1: CLEAN
    # ------------------------------------------------------------------
    print("\n##### STEP 1: CLEAN #####\n")
    cleaner = RetailCleaner(str(RAW_FILE)).run()
    cleaner.save(DATA_DIR)

    # ------------------------------------------------------------------
    # STEP 2: EDA
    # ------------------------------------------------------------------
    print("\n##### STEP 2: EDA #####\n")
    eda = RetailEDA(cleaner.sales).run()


    # ------------------------------------------------------------------
    # STEP 3: RFM  (build retail_rfm.py next)
    # ------------------------------------------------------------------
    print("\n##### STEP 3: RFM #####\n")
    rfm = RFMBuilder(cleaner.customer_sales).run()
    rfm.save(DATA_DIR)

    # ------------------------------------------------------------------
    # STEP 3: SEGMENTATION
    # ------------------------------------------------------------------
    print("\n##### STEP 3: SEGMENTATION #####\n")
    seg = CustomerSegmenter(rfm.features, cleaner.customer_sales).run()
    seg.fit(5).profile()
    seg.fit(6).profile()
    seg.fit(7).profile()

    seg.run(k=6).save(DATA_DIR)
    
    # ------------------------------------------------------------------
    # STEP 4: BASKET ANALYSIS
    # ------------------------------------------------------------------
    print("\n##### STEP 4: BASKET ANALYSIS #####\n")
    basket = BasketAnalyser(cleaner.sales, min_support=0.01).run()
    
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()

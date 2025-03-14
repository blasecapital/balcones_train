# test_pipeline_0.py


import shutil
import os
import unittest
from unittest.mock import patch
from itertools import islice

import sqlite3
import pandas as pd

from data_preparation.components.create_features import CreateFeatures
from data_preparation.components.create_targets import CreateTargets
from training.components.clean_raw_data import CleanRawData
from training.components.prep_data import PrepData
from training.components.train import Train
from evaluation.components.evaluate import Eval


def start():
    # Copy config.env to /workspace
    source_path = "/workspace/tests/integration_tests/fx_classify/pipeline_0/config.env"
    destination_path = "/workspace/config.env"

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    try:
        shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
        print(f"File copied successfully from {source_path} to {destination_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied: Unable to copy to {destination_path}")
    except Exception as e:
        raise Exception(f"Error copying file: {e}")
    

def end():
    # Delete train and clean databases after completion
    train_db = '/workspace/tests/integration_tests/fx_classify/fixtures/fx_classify_test_train_data.db'
    clean_db = '/workspace/tests/integration_tests/fx_classify/fixtures/fx_classify_test_train_data_clean.db'
    if os.path.exists(train_db):
        os.remove(train_db)
    if os.path.exists(clean_db):
        os.remove(clean_db)


class TestFxClassify0(unittest.TestCase):

    def test_start(self):
        start()

    def test_CreateFeatures(self):
        cf = CreateFeatures()
        cf.calculate_and_store_features()

    def test_CreateTargets(self):
        ct = CreateTargets()
        ct.calculate_and_store_targets()

    def test_train_db(self):
        db_path = "/workspace/tests/integration_tests/fx_classify/fixtures/fx_classify_test_train_data.db"
        conn = sqlite3.connect(db_path)
        target_query = """
        SELECT * FROM targets
        ORDER BY date, pair
        LIMIT 5
        """
        targets = pd.read_sql(target_query, conn)
        targets = targets.astype({
            "date": "string",
            "pair": "string",
            "target": "string",
            "hours_passed": "string",
            "buy_sl_time": "string",
            "sell_sl_time": "string"
        })

        hourly_query = """
        SELECT date, pair, open_standard_1, high_standard_1, low_standard_1, close_standard_1 FROM test_hourly_feature_data
        WHERE date = '2025-02-03 08:00:00'
        """
        hourly = pd.read_sql(hourly_query, conn)
        hourly = hourly.astype({
            "date": "string",
            "pair": "string",
            "open_standard_1": "string",
            "high_standard_1": "string",
            "low_standard_1": "string",
            "close_standard_1": "string"
        })

        tech_query = """
        SELECT * FROM test_technical_feature_data
        WHERE date = '2025-02-03 08:00:00'
        """
        tech = pd.read_sql(tech_query, conn)
        tech = tech.astype({
            "date": "string",
            "pair": "string",
            "high_vol": "string",
            "sma_roc": "string",
            "close_to_std_dev": "string"
        })
        conn.close()

        expected_targets = pd.DataFrame({
            "date": [
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00",
                "2025-01-01 00:00:00"
            ],
            "pair": ["AUDUSD", "CHFJPY", "EURUSD", "USDJPY", "USDMXN"],
            "target": ["loss", "loss", "loss", "loss", "loss"],
            "hours_passed": [1.0, 1.0, 1.0, 1.0, 2.0],
            "buy_sl_time": [1.0, 1.0, 1.0, 1.0, 2.0],
            "sell_sl_time": [1.0, 1.0, 1.0, 1.0, 1.0]
        }).astype({
            "date": "string",
            "pair": "string",
            "target": "string",
            "hours_passed": "string",
            "buy_sl_time": "string",
            "sell_sl_time": "string"
        })

        expected_hourly = pd.DataFrame({
            "date": [
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
            ],
            "pair": ["AUDUSD", "CHFJPY", "EURUSD", "USDJPY", "USDMXN"],
            "open_standard_1": [0.99734, 0.99878, 0.99676, 0.99822, 1.00322],
            "high_standard_1": [0.99803, 0.99952, 0.99754, 1.00155, 1.00609],
            "low_standard_1": [0.99703, 0.99866, 0.99662, 0.99611, 0.99998],
            "close_standard_1": [0.99756, 0.99887, 0.99669, 0.99993, 1.00497]
        }).astype({
            "date": "string",
            "pair": "string",
            "open_standard_1": "string",
            "high_standard_1": "string",
            "low_standard_1": "string",
            "close_standard_1": "string"
        })

        expected_tech = pd.DataFrame({
            "date": [
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
                "2025-02-03 08:00:00",
            ],
            "pair": ["AUDUSD", "CHFJPY", "EURUSD", "USDJPY", "USDMXN"],
            "high_vol": [0, 1, 1, 1, 1],
            "sma_roc": [0.0000665632977806158,
                        -0.000640382307154887,
                        -0.00023106484409193,
                        0.000150175333526161,
                        0.000801112226175249],
            "close_to_std_dev": [-1.25777660977386,
                                0.83790350653372,
                                -0.495739238016207,
                                -0.835592985852584,
                                -1.41764776199103]
        }).astype({
            "date": "string",
            "pair": "string",
            "high_vol": "string",
            "sma_roc": "string",
            "close_to_std_dev": "string"
        })
        
        pd.testing.assert_frame_equal(targets, expected_targets)
        pd.testing.assert_frame_equal(hourly, expected_hourly)
        pd.testing.assert_frame_equal(tech, expected_tech)

    @patch("matplotlib.pyplot.show")
    def test_CleanRawData(self, mock_show):
        clean = CleanRawData()

        # Test inspect_data
        inspect_cases = [
            {"data_keys": ["hourly_features"], "plot_features": True, "plot_mode": "rate"},
            {"data_keys": ["tech_features"], "describe_features": True, "plot_features": True, "plot_mode": "stat"},
            {"data_keys": ["targets"], "describe_targets": True, "target_type": "cat"}
        ]

        for args in inspect_cases:
            with self.subTest(args=args):
                clean.inspect_data(**args)

                if not args.get("plot_features", False):
                    continue
                mock_show.assert_called()

        # Test filter_keys
        clean.clean(data_keys=[])

        hourly_keys = '/workspace/tests/integration_tests/fx_classify/pipeline_0/bad_keys/hourly_features_bad_keys.txt'
        target_keys = '/workspace/tests/integration_tests/fx_classify/pipeline_0/bad_keys/targets_bad_keys.txt'

        with open(hourly_keys, "r") as file:
            hourly_five = list(islice(file, 5))
        with open(target_keys, "r") as file:
            target_five = list(islice(file, 5))
        
        expected_hourly = ["('AUDUSD', '2025-01-01 00:00:00')\n", 
                           "('CHFJPY', '2025-01-01 00:00:00')\n", 
                           "('EURUSD', '2025-01-01 00:00:00')\n", 
                           "('USDJPY', '2025-01-01 00:00:00')\n", 
                           "('USDMXN', '2025-01-01 00:00:00')\n"]
        expected_target = ["('EURUSD', '2025-02-08 13:00:00')\n", 
                           "('EURUSD', '2025-02-08 14:00:00')\n", 
                           "('AUDUSD', '2025-02-08 23:00:00')\n", 
                           "('EURUSD', '2025-02-09 09:00:00')\n", 
                           "('CHFJPY', '2025-02-10 02:00:00')\n"]

        self.assertEqual(hourly_five, expected_hourly)
        self.assertEqual(target_five, expected_target)

        # Test align
        clean.align()
        expected_result = 4888
        
        conn = sqlite3.connect('/workspace/tests/integration_tests/fx_classify/fixtures/fx_classify_test_train_data_clean.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(date) FROM targets")
        target_len = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(date) FROM test_hourly_feature_data")
        hourly_len = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(date) FROM test_technical_feature_data")
        tech_len = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(expected_result, target_len)
        self.assertEqual(expected_result, hourly_len)
        self.assertEqual(expected_result, tech_len)

    def test_PrepData(self):
        prep = PrepData()
        prep.engineer("all")

        conn = sqlite3.connect('/workspace/tests/integration_tests/fx_classify/fixtures/fx_classify_test_train_data_clean.db')

        targets = pd.read_sql("SELECT target FROM targets ORDER BY date, pair LIMIT 5", conn)
        targets = targets.astype({"target": "string"})
        expected_targets = pd.DataFrame({"target": [0, 0, 0, 0, 2]}).astype({"target": "string"})
        pd.testing.assert_frame_equal(targets, expected_targets)

        engineered = pd.read_sql("SELECT * FROM test_engineered_feature_data ORDER BY date, pair LIMIT 5", conn)
        engineered = engineered.astype({
            "date": "string",
            "pair": "string",
            "hour_sin": "string",
            "hour_cos": "string",
            "pair_AUDUSD": "string",
            "pair_CHFJPY": "string",
            "pair_EURUSD": "string",
            "pair_USDJPY": "string",
            "pair_USDMXN": "string"
        })
        expected_engineered = pd.DataFrame({
            "date": ["2025-01-01 12:00:00",
                    "2025-01-01 12:00:00",
                    "2025-01-01 12:00:00",
                    "2025-01-01 12:00:00",
                    "2025-01-01 12:00:00"],
            "pair": ["AUDUSD", "CHFJPY", "EURUSD", "USDJPY", "USDMXN"],
            "hour_sin": [1.2246467991473532e-16, 1.2246467991473532e-16, 1.2246467991473532e-16, 1.2246467991473532e-16, 1.2246467991473532e-16],
            "hour_cos": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "pair_AUDUSD": [1, 0, 0, 0, 0],
            "pair_CHFJPY": [0, 1, 0, 0, 0],
            "pair_EURUSD": [0, 0, 1, 0, 0],
            "pair_USDJPY": [0, 0, 0, 1, 0],
            "pair_USDMXN": [0, 0, 0, 0, 1],
        }).astype({
            "date": "string",
            "pair": "string",
            "hour_sin": "string",
            "hour_cos": "string",
            "pair_AUDUSD": "string",
            "pair_CHFJPY": "string",
            "pair_EURUSD": "string",
            "pair_USDJPY": "string",
            "pair_USDMXN": "string"
        })
        pd.testing.assert_frame_equal(engineered, expected_engineered)

        prep.feature_scaling["test_features"] = (
            "CLEAN_FEATURE_DATABASE",
            ['test_hourly_feature_data']
            )
        prep.scale()
        prep.feature_scaling["test_features"] = (
            "CLEAN_FEATURE_DATABASE",
            ['test_technical_feature_data']
            )
        prep.scale()

        prep.save_batches()
        prep.validate_record()

    def test_Train(self):
        train = Train()
        train.train_models()        

    @patch("matplotlib.pyplot.show")
    def test_Eval(self, mock_show):
        eval = Eval()
        eval.predict_and_store(mode="full")

        conn = sqlite3.connect('/workspace/tests/integration_tests/fx_classify/pipeline_0/predictions/14_03_2025_23_02_54_epoch4.db')
        pred = pd.read_sql("""SELECT confidence FROM predictions ORDER BY date, pair LIMIT 5""", conn).astype({"confidence": "string"})
        expected_pred = pd.DataFrame({
            "confidence": [
                0.3798828125,
                0.387451171875,
                0.39453125,
                0.458251953125,
                0.473388671875,
            ]
        }).astype({"confidence": "string"})
        pd.testing.assert_frame_equal(pred, expected_pred)

        eval.explain()
        eval.report_metrics()
        eval.report_metrics(display_mode="convert")
        eval.report_calibration()
        eval.report_calibration(mode="convert")
        eval.report_candidates(
            mode="convert",
            accuracy_threshold=.35,
            volume=10
        )

    def test_end(self):
        end()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFxClassify0("test_start"))
    suite.addTest(TestFxClassify0("test_CreateFeatures"))
    suite.addTest(TestFxClassify0("test_CreateTargets"))
    suite.addTest(TestFxClassify0("test_train_db"))
    suite.addTest(TestFxClassify0("test_CleanRawData"))
    suite.addTest(TestFxClassify0("test_PrepData"))
    suite.addTest(TestFxClassify0("test_Train"))
    suite.addTest(TestFxClassify0("test_Eval"))

    #suite.addTest(TestFxClassify0("test_end"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

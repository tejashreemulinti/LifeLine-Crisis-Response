import json
import os

class ReporterAgent:
    def save_report(self, report):
        os.makedirs("reports", exist_ok=True)

        with open("reports/report.json", "w") as f:
            json.dump(report, f, indent=4)

        return "Report saved to reports/report.json"

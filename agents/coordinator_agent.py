class CoordinatorAgent:
    def combine(self, detection, verification, geo):
        return {
            "category": detection,
            "verification": verification,
            "location_info": geo
        }

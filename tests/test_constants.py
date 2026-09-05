"""Unit test suite verifying spatial, orbital, and geomagnetic normalization transformations in constants.py."""

import unittest
import numpy as np
import constants


class TestConstants(unittest.TestCase):
    """Validates physical variable normalization transformations against analytical bounds."""

    def test_normalizations_dict(self):
        """Tests that all required coordinate and geomagnetic features are present and scaled properly."""
        norms = constants.NORMALIZATIONS

        # 1. Altitude
        self.assertIn("Altitude", norms)
        self.assertAlmostEqual(float(norms["Altitude"](4000)), 0.0)
        self.assertAlmostEqual(float(norms["Altitude"](8000)), 1.0)
        self.assertAlmostEqual(float(norms["Altitude"](0)), -1.0)

        # 2. Latitudes (GCLAT, ILAT, GLAT, XXLAT)
        for lat_key in ["GCLAT", "ILAT", "GLAT", "XXLAT"]:
            self.assertIn(lat_key, norms)
            self.assertAlmostEqual(float(norms[lat_key](0.0)), 0.0)
            self.assertAlmostEqual(float(norms[lat_key](90.0)), 1.0)
            self.assertAlmostEqual(float(norms[lat_key](-90.0)), -1.0)

        # 3. Longitudes (GCLON, XXLON)
        for lon_key in ["GCLON", "XXLON"]:
            self.assertIn(lon_key, norms)
            self.assertAlmostEqual(float(norms[lon_key](0.0)), 0.0)
            self.assertAlmostEqual(float(norms[lon_key](90.0)), 1.0)
            self.assertAlmostEqual(float(norms[lon_key](180.0)), 0.0, places=5)

        # 4. Magnetic Local Time (GMLT)
        self.assertIn("GMLT", norms)
        self.assertAlmostEqual(float(norms["GMLT"](0.0)), 0.0)
        self.assertAlmostEqual(float(norms["GMLT"](6.0)), 1.0)
        self.assertAlmostEqual(float(norms["GMLT"](12.0)), 0.0, places=5)
        self.assertAlmostEqual(float(norms["GMLT"](18.0)), -1.0)

        # 5. Planetary Kp index
        self.assertIn("Kp_index", norms)
        self.assertAlmostEqual(float(norms["Kp_index"](0)), -1.0)
        self.assertAlmostEqual(float(norms["Kp_index"](45)), 0.0)
        self.assertAlmostEqual(float(norms["Kp_index"](90)), 1.0)

        # 6. Orthogonal Cosine Coordinates (resolving east/west & noon/midnight ambiguities)
        for cos_key in ["GCLON_cos", "XXLON_cos"]:
            self.assertIn(cos_key, norms)
            self.assertAlmostEqual(float(norms[cos_key](0.0)), 1.0)
            self.assertAlmostEqual(float(norms[cos_key](90.0)), 0.0, places=5)
            self.assertAlmostEqual(float(norms[cos_key](180.0)), -1.0)

        self.assertIn("GMLT_cos", norms)
        self.assertAlmostEqual(float(norms["GMLT_cos"](0.0)), 1.0)
        self.assertAlmostEqual(float(norms["GMLT_cos"](6.0)), 0.0, places=5)
        self.assertAlmostEqual(float(norms["GMLT_cos"](12.0)), -1.0)
        self.assertAlmostEqual(float(norms["GMLT_cos"](18.0)), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

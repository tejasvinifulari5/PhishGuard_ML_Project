import unittest

from phishguard_pipeline import (
    LEXICAL_FEATURE_COLUMNS,
    build_lexical_frame,
    extract_lexical_features,
    normalize_url,
)


class URLFeatureTests(unittest.TestCase):
    def test_https_url_returns_all_ordered_numeric_features(self):
        features = extract_lexical_features("https://example.com/login")

        self.assertEqual(list(features), LEXICAL_FEATURE_COLUMNS)
        self.assertTrue(all(isinstance(value, (int, float)) for value in features.values()))

    def test_scheme_less_url_gets_https(self):
        self.assertEqual(
            normalize_url("example.com/login"),
            "https://example.com/login",
        )
        self.assertEqual(extract_lexical_features("example.com")["has_https"], 1)

    def test_ip_address_is_detected(self):
        features = extract_lexical_features("http://127.0.0.1/login")

        self.assertEqual(features["has_ip_address"], 1)

    def test_query_and_percent_encoding_are_preserved(self):
        url = "https://example.com/search?q=a%2Fb&next=%3F"

        self.assertEqual(normalize_url(url), url)
        self.assertEqual(
            extract_lexical_features(url)["query_length"],
            len("q=a%2Fb&next=%3F"),
        )

    def test_unicode_and_punycode_hosts_are_supported(self):
        unicode_url = "https://b\u00fccher.example/path"
        punycode_url = "https://xn--bcher-kva.example/path"

        self.assertEqual(normalize_url(unicode_url), unicode_url)
        self.assertEqual(extract_lexical_features(unicode_url)["has_punycode"], 0)
        self.assertEqual(extract_lexical_features(punycode_url)["has_punycode"], 1)

    def test_malformed_brackets_and_ports_are_rejected(self):
        for url in ("http://[malformed", "http://example.com:99999"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    normalize_url(url)
                with self.assertRaises(ValueError):
                    extract_lexical_features(url)

    def test_lexical_frame_has_expected_feature_shape(self):
        frame = build_lexical_frame(["https://example.com", "example.org/login"])

        self.assertEqual(frame.shape, (2, len(LEXICAL_FEATURE_COLUMNS)))
        self.assertEqual(list(frame.columns), LEXICAL_FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()

# SEO, social previews and verified screenshot policy

Public pages have route-specific canonical, title, description, Open Graph/Twitter image and JSON-LD WebPage metadata. The social images below are cropped from actual deployed app screenshots, with source hashes verified against existing 14 October 2026 screenshot manifests. The images are not model-performance evidence.

| Public URL | Social JPEG | Verified source PNG | Source SHA-256 | File size |
| --- | --- | --- | --- | ---: |
| https://smart-factory-app.onrender.com/ | /assets/seo/factory-social.jpg | docs/assets/smart-factory-dashboard.png | a486686cea998008ad7592dd0a70f7145ce32cdbf5aa80af143e8a0a4579b2c2 | 128625 B |
| https://smart-factory-app.onrender.com/case-study | /assets/seo/maintenance-social.jpg | docs/assets/smart-factory-maintenance.png | 9b4fe16ec9b53860ebcb8d0ce6fdd5cafbc83ecf21ce4f4f1210364de8236b45 | 158379 B |
| https://smart-factory-app.onrender.com/rul-case-study | /assets/seo/rul-social.jpg | docs/assets/smart-factory-rul.png | 13638eb5a768b8b4d05bcdd1bfcccb2ae1506f978f953ac2b5600fb22cbd2905 | 136358 B |

## Interpretation and maintenance

- Fraud Analytics displays historical governed BigQuery reports, not continuously observed transactions.
- Smart Factory and Visual QC operator dashboards are simulations, not actual equipment feeds.
- A model benchmark screenshot must not be interpreted as a deployment or plant-performance result.
- Cropping and JPEG compression are image-delivery changes; no metrics or visual content were invented.
- Robots/sitemap endpoints advertise public page URLs only. Robots policy is not API authorization.
- External search indexing and social card cache refresh times are controlled by third-party crawlers.
- Regenerate preview assets only from provenance-verified screenshots and re-run the local validation script.

## Verification

Run: python scripts/check_seo_assets.py
Then verify public URLs plus /robots.txt and /sitemap.xml after release.

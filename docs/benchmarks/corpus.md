# Benchmark documents — the exact 125 OmniDocBench pages used for every number

Regenerate deterministically with `scripts/eval/omnidoc_subset_n.py --n 125` (it writes `OmniDocBench_subset125.json`; the `/tmp` path in older runs is just its default output dir).
Columns: filename · source · language · #GT-tables · #GT-formulas · #GT-text-blocks. All 125 pages contain a table and/or a formula (that is the point of this stratified subset).

| # | document (image_path) | source | language | tbl | fml | text |
|---|---|---|---|---|---|---|
| 1 | `PPT_CalculusReview_page_014.png` | PPT2PDF | english | 0 | 2 | 2 |
| 2 | `PPT_fundamental_theorem_of_calculus___page_013.png` | PPT2PDF | english | 0 | 1 | 2 |
| 3 | `PPT_lay_linalg5_01_05_page_004.png` | PPT2PDF | english | 0 | 2 | 3 |
| 4 | `PPT_linear-algebra primer_page_008.png` | PPT2PDF | english | 0 | 1 | 6 |
| 5 | `page-00b6ac57-4466-4eb0-937d-bb29a44fa0d3.png` | PPT2PDF | en_ch_mixed | 1 | 0 | 34 |
| 6 | `page-31fb2a53-5b32-40f6-9db1-52b357f3201f.png` | PPT2PDF | en_ch_mixed | 1 | 0 | 0 |
| 7 | `page-39551bb3-1b65-4562-b258-1bc97898cdf9.png` | PPT2PDF | simplified_chinese | 0 | 5 | 6 |
| 8 | `page-aee8f1d9-7240-4517-b615-389594556d57.png` | PPT2PDF | simplified_chinese | 0 | 3 | 1 |
| 9 | `page-ba4443e6-f432-435f-89e7-ddf0a571d2cd.png` | PPT2PDF | simplified_chinese | 0 | 3 | 6 |
| 10 | `yanbaopptmerge_yanbaoPPT_2460.jpg` | PPT2PDF | simplified_chinese | 0 | 7 | 3 |
| 11 | `yanbaor2_yanbaoPPT_916.jpg` | PPT2PDF | simplified_chinese | 1 | 0 | 1 |
| 12 | `docstructbench_llm-raw-scihub-o.O-ceat.200407001.pdf_3.jpg` | academic_literature | english | 1 | 7 | 12 |
| 13 | `docstructbench_llm-raw-scihub-o.O-ceat.200600410.pdf_5.jpg` | academic_literature | english | 2 | 1 | 3 |
| 14 | `docstructbench_llm-raw-scihub-o.O-j.jcp.2010.12.006.pdf_14.jpg` | academic_literature | english | 0 | 4 | 4 |
| 15 | `docstructbench_llm-raw-scihub-o.O-j.physletb.2004.11.060.pdf_2.jpg` | academic_literature | english | 0 | 7 | 14 |
| 16 | `docstructbench_llm-raw-scihub-o.O-j.snb.2004.05.025.pdf_5.jpg` | academic_literature | english | 0 | 2 | 2 |
| 17 | `docstructbench_llm-raw-scihub-o.O-s00128-008-9367-z.pdf_4.jpg` | academic_literature | english | 2 | 3 | 4 |
| 18 | `docstructbench_llm-raw-scihub-o.O-s00348-006-0117-x.pdf_11.jpg` | academic_literature | english | 0 | 4 | 6 |
| 19 | `docstructbench_llm-raw-the-eye-o.O-1995_2418.pdf_1.jpg` | academic_literature | english | 1 | 0 | 20 |
| 20 | `page-28657a85-95cc-4bfa-8156-cc98518a0b4c.png` | academic_literature | simplified_chinese | 1 | 1 | 5 |
| 21 | `page-3ce15378-6ffd-4f3c-9473-8b897451c770.png` | academic_literature | en_ch_mixed | 1 | 0 | 0 |
| 22 | `page-404e0471-bd71-426a-8e5a-81b8327f88e5.png` | academic_literature | english | 0 | 19 | 16 |
| 23 | `page-8d707810-de48-43ba-81a7-b976918e7be2.png` | academic_literature | en_ch_mixed | 1 | 0 | 0 |
| 24 | `page-9f6bc620-1983-4503-bb0f-f9c7949ccb77.png` | academic_literature | simplified_chinese | 1 | 5 | 4 |
| 25 | `page-f108020f-edc5-4a27-85d9-d9d9c5d67309.png` | academic_literature | english | 0 | 17 | 16 |
| 26 | `page-f9583127-1277-4424-be9d-35430e0aedd6.png` | academic_literature | english | 1 | 2 | 12 |
| 27 | `scihub_j.conbuildmat.2019.117698.pdf_5.jpg` | academic_literature | english | 0 | 1 | 6 |
| 28 | `book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0100.png` | book | english | 2 | 7 | 11 |
| 29 | `book_zh_CNASGL0262018_extracted_page_82.png` | book | simplified_chinese | 1 | 8 | 7 |
| 30 | `book_zh_CNASGL082006_extracted_page_50.png` | book | simplified_chinese | 1 | 4 | 2 |
| 31 | `book_zh_DLT10902008_extracted_page_8.png` | book | simplified_chinese | 1 | 5 | 7 |
| 32 | `book_zh_GB115041989_extracted_page_9.png` | book | simplified_chinese | 1 | 2 | 14 |
| 33 | `book_zh_GB14536.11998_extracted_page_145.png` | book | simplified_chinese | 1 | 1 | 16 |
| 34 | `book_zh_HGT51022016_extracted_page_12.png` | book | simplified_chinese | 1 | 1 | 10 |
| 35 | `book_zh_JGJ1022003_extracted_page_27.png` | book | simplified_chinese | 0 | 6 | 15 |
| 36 | `book_zh_JJD10031991_extracted_page_6.png` | book | simplified_chinese | 0 | 4 | 14 |
| 37 | `book_zh_TB100782001_extracted_page_21.png` | book | simplified_chinese | 0 | 6 | 11 |
| 38 | `book_zh_YDT10512018_extracted_page_30.png` | book | simplified_chinese | 1 | 3 | 5 |
| 39 | `docstructbench_dianzishu_zhongwenzaixian-o.O-61520788.pdf_391.jpg` | book | simplified_chinese | 1 | 1 | 12 |
| 40 | `docstructbench_dianzishu_zhongwenzaixian-o.O-61569751.pdf_155.jpg` | book | simplified_chinese | 1 | 1 | 10 |
| 41 | `page-423cdc11-af52-4e41-8feb-a40741b15855.png` | book | traditional_chinese | 1 | 0 | 4 |
| 42 | `page-573c437e-c309-4483-a038-ef2f440b104a.png` | book | english | 1 | 1 | 3 |
| 43 | `page-773aa2c0-3413-41d0-b019-a573da251455.png` | book | en_ch_mixed | 0 | 1 | 2 |
| 44 | `page-ab441454-71f1-4900-87c1-d71d5602261e.png` | book | en_ch_mixed | 2 | 0 | 0 |
| 45 | `page-eb7308e0-1395-4444-b6a5-9d33cc3a03b2.png` | book | en_ch_mixed | 1 | 0 | 0 |
| 46 | `yanbaopptmerge_3aa9b6677e17fd012b0a7a230b9f18db.pdf_1327.jpg` | book | english | 1 | 6 | 15 |
| 47 | `yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg` | book | english | 0 | 72 | 5 |
| 48 | `yanbaopptmerge_abef2a4978ae4d13e931f0392502bd40.pdf_1287.jpg` | book | english | 0 | 70 | 41 |
| 49 | `yanbaopptmerge_d4fc7cba428625974e93183edfccea73.pdf_89.jpg` | book | english | 1 | 1 | 7 |
| 50 | `color_textbook_教材全解1+1二年级上册英语上海牛津版_page_058.png` | colorful_textbook | en_ch_mixed | 2 | 0 | 11 |
| 51 | `color_textbook_教材全解1+1二年级下册英语上海牛津版_page_006.png` | colorful_textbook | en_ch_mixed | 1 | 0 | 9 |
| 52 | `color_textbook_教材全解1+1二年级下册英语上海牛津版_page_066.png` | colorful_textbook | en_ch_mixed | 1 | 0 | 10 |
| 53 | `color_textbook_教材全解1+1二年级下册英语上海牛津版_page_081.png` | colorful_textbook | en_ch_mixed | 1 | 0 | 7 |
| 54 | `docstructbench_enbook-zlib-o.O-17761417.pdf_894.jpg` | colorful_textbook | english | 1 | 0 | 3 |
| 55 | `jiaocaineedrop_jiaocai_needrop_en_1118.jpg` | colorful_textbook | en_ch_mixed | 1 | 0 | 0 |
| 56 | `jiaocaineedrop_jiaocai_needrop_en_1253.jpg` | colorful_textbook | simplified_chinese | 0 | 2 | 9 |
| 57 | `jiaocaineedrop_jiaocai_needrop_en_1496.jpg` | colorful_textbook | simplified_chinese | 0 | 12 | 2 |
| 58 | `jiaocaineedrop_jiaocai_needrop_en_1910.jpg` | colorful_textbook | simplified_chinese | 1 | 2 | 14 |
| 59 | `jiaocaineedrop_jiaocai_needrop_en_250.jpg` | colorful_textbook | simplified_chinese | 0 | 2 | 9 |
| 60 | `jiaocaineedrop_jiaocai_needrop_en_541.jpg` | colorful_textbook | simplified_chinese | 0 | 11 | 3 |
| 61 | `jiaocaineedrop_jiaocai_needrop_en_546.jpg` | colorful_textbook | simplified_chinese | 0 | 4 | 3 |
| 62 | `jiaocaineedrop_jiaocai_needrop_en_913.jpg` | colorful_textbook | simplified_chinese | 0 | 6 | 11 |
| 63 | `page-67013be9-58e5-4842-809d-7a3c1fc91fc7.png` | colorful_textbook | simplified_chinese | 0 | 2 | 11 |
| 64 | `page-8f6792bd-b5e4-435e-b1b0-1b2daa3f7234.png` | colorful_textbook | simplified_chinese | 0 | 2 | 15 |
| 65 | `page-bea34248-acae-4578-9ff2-3149a84d7c38.png` | colorful_textbook | simplified_chinese | 0 | 2 | 12 |
| 66 | `exam_paper_2018年广西北海、钦州、南宁、来宾、崇左、防城港、北部湾经济区中考英语试题（空白卷）_page_003.png` | exam_paper | en_ch_mixed | 1 | 0 | 6 |
| 67 | `exam_paper_en-file-putnam-archive_1991_Problems_1991_page_001.png` | exam_paper | english | 0 | 7 | 24 |
| 68 | `exam_paper_en-file-putnam-archive_1993_Problems_1993_page_001.png` | exam_paper | english | 0 | 6 | 16 |
| 69 | `exam_paper_en-file-putnam-archive_1994_Problems_1994_page_001.png` | exam_paper | english | 0 | 6 | 18 |
| 70 | `exam_paper_en-file-putnam-archive_1995_Solutions_1995s_page_002.png` | exam_paper | english | 0 | 5 | 16 |
| 71 | `exam_paper_en-file-putnam-archive_1996_Problems_1996_page_001.png` | exam_paper | english | 0 | 6 | 17 |
| 72 | `exam_paper_en-file-putnam-archive_1996_Solutions_1996s_page_001.png` | exam_paper | english | 0 | 2 | 15 |
| 73 | `exam_paper_en-file-putnam-archive_1996_Solutions_1996s_page_002.png` | exam_paper | english | 0 | 8 | 18 |
| 74 | `exam_paper_en-file-putnam-archive_1996_Solutions_1996s_page_003.png` | exam_paper | english | 0 | 8 | 16 |
| 75 | `exam_paper_en-file-putnam-archive_1997_Problems_1997_page_001.png` | exam_paper | english | 0 | 7 | 16 |
| 76 | `exam_paper_en-file-putnam-archive_1997_Solutions_1997s_page_001.png` | exam_paper | english | 0 | 14 | 21 |
| 77 | `exam_paper_en-file-putnam-archive_1997_Solutions_1997s_page_002.png` | exam_paper | english | 0 | 9 | 17 |
| 78 | `exam_paper_en-file-putnam-archive_1999_Solutions_1999s_page_001.png` | exam_paper | english | 0 | 9 | 20 |
| 79 | `exam_paper_en-file-putnam-archive_2000_Problems_2000_page_001.png` | exam_paper | english | 0 | 3 | 15 |
| 80 | `jiaocai_71434495.pdf_0.jpg` | exam_paper | simplified_chinese | 1 | 13 | 29 |
| 81 | `jiaocaineedrop_20608808.pdf_0.jpg` | exam_paper | simplified_chinese | 1 | 5 | 17 |
| 82 | `jiaocaineedrop_33137159.pdf_3.jpg` | exam_paper | simplified_chinese | 1 | 8 | 10 |
| 83 | `jiaocaineedrop_38247658.pdf_0.jpg` | exam_paper | simplified_chinese | 1 | 3 | 19 |
| 84 | `jiaocaineedrop_42351289.pdf_1.jpg` | exam_paper | simplified_chinese | 1 | 2 | 19 |
| 85 | `jiaocaineedrop_jiaocai_needrop_en_1349.jpg` | exam_paper | simplified_chinese | 1 | 1 | 18 |
| 86 | `jiaocaineedrop_jiaocai_needrop_en_2124.jpg` | exam_paper | en_ch_mixed | 1 | 0 | 15 |
| 87 | `jiaocaineedrop_jiaocai_needrop_en_2211.jpg` | exam_paper | simplified_chinese | 1 | 1 | 30 |
| 88 | `jiaocaineedrop_jiaocai_needrop_en_2901.jpg` | exam_paper | en_ch_mixed | 2 | 0 | 14 |
| 89 | `jiaocaineedrop_jiaocai_needrop_en_467.jpg` | exam_paper | en_ch_mixed | 1 | 0 | 10 |
| 90 | `jiaocaineedrop_jiaocai_needrop_en_482.jpg` | exam_paper | en_ch_mixed | 1 | 0 | 17 |
| 91 | `jiaocaineedrop_jiaocai_needrop_en_999.jpg` | exam_paper | simplified_chinese | 1 | 2 | 25 |
| 92 | `page-4319d401-c9e8-4326-9869-7572cf2e0e96.png` | exam_paper | traditional_chinese | 1 | 0 | 5 |
| 93 | `docstructbench_dianzishu_zhongwenzaixian-o.O-61520814.pdf_185.jpg` | magazine | simplified_chinese | 1 | 0 | 5 |
| 94 | `docstructbench_llm-raw-the-eye-o.O-TFT-Traveller.pdf_6.jpg` | magazine | english | 3 | 0 | 0 |
| 95 | `jiaocaineedrop_chap02.pdf_16.jpg` | magazine | english | 1 | 0 | 18 |
| 96 | `magazine_TheEconomist.2024.02.24_page_076.png` | magazine | english | 5 | 0 | 0 |
| 97 | `yanbaor2_3e1be78252e2fdfe1adf12bba38ec2a7b30699e152d61269aa6e5827f5adcc35.pdf_13.jpg` | magazine | simplified_chinese | 1 | 0 | 1 |
| 98 | `newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_1.jpg` | newspaper | english | 2 | 0 | 18 |
| 99 | `newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_18.jpg` | newspaper | english | 2 | 0 | 16 |
| 100 | `newspaper_2a6b4fa088699701a6fa9ccecfb5c25d_2.jpg` | newspaper | english | 1 | 0 | 12 |
| 101 | `newspaper_Chicago Tribune_0801@magazinesclubnew_page_019.png` | newspaper | english | 13 | 0 | 36 |
| 102 | `newspaper_Daily Star 2025-1-8@magazinesclubnew_page_060.png` | newspaper | english | 2 | 0 | 0 |
| 103 | `newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_014.png` | newspaper | english | 5 | 0 | 20 |
| 104 | `newspaper_The Guardian UK_0801@magazinesclubnew_page_052.png` | newspaper | english | 2 | 0 | 7 |
| 105 | `newspaper_The Times UK_0801@magazinesclubnew_page_031.png` | newspaper | english | 1 | 0 | 42 |
| 106 | `newspaper_fe5ed29024932fad071afc53807b16ba_2.jpg` | newspaper | english | 2 | 0 | 14 |
| 107 | `notes_1ba14cb325bc448f7201b20502ecf2b5_16.jpg` | note | simplified_chinese | 0 | 3 | 3 |
| 108 | `notes_9e951846094758afac08c620144e3a76_10.jpg` | note | simplified_chinese | 0 | 6 | 5 |
| 109 | `notes_9e951846094758afac08c620144e3a76_14.jpg` | note | simplified_chinese | 0 | 3 | 8 |
| 110 | `notes_9e951846094758afac08c620144e3a76_15.jpg` | note | simplified_chinese | 0 | 5 | 7 |
| 111 | `notes_9e951846094758afac08c620144e3a76_16.jpg` | note | simplified_chinese | 0 | 5 | 8 |
| 112 | `notes_f7f010b78016aeebd76e56d9283eb67f_49.jpg` | note | en_ch_mixed | 1 | 0 | 11 |
| 113 | `notes_f7f010b78016aeebd76e56d9283eb67f_50.jpg` | note | en_ch_mixed | 2 | 0 | 1 |
| 114 | `eastmoney_1746461e6e00efee224a9209974b4bb6de3b11e339e4d86c41569eeddcc3c20e.pdf_8.jpg` | research_report | simplified_chinese | 1 | 0 | 2 |
| 115 | `eastmoney_1885ca41425d245551f3482457304f78b48186bff625fa91e675eaf6bba5229f.pdf_0.jpg` | research_report | simplified_chinese | 3 | 0 | 7 |
| 116 | `eastmoney_34d623c64d12b5f02ffe4bff74f464b368270b1d6930192876bfe353d8fd6c30.pdf_0.jpg` | research_report | simplified_chinese | 1 | 0 | 7 |
| 117 | `eastmoney_62b4149b1612ce28d20f26cd5c5b2e18f80b26fca6e4452e090376a2fe72eae3.pdf_0.jpg` | research_report | simplified_chinese | 2 | 0 | 13 |
| 118 | `eastmoney_66eea274d39b939da0f10253d279e119d87646f10fd21b3942eaf6c5d93b8134.pdf_0.jpg` | research_report | simplified_chinese | 2 | 0 | 9 |
| 119 | `eastmoney_944c06af0b176ef718ee34d3affee40554920f76210dfbf007972fe9c39074fc.pdf_0.jpg` | research_report | simplified_chinese | 2 | 0 | 10 |
| 120 | `eastmoney_a2542ccf95ec3bb51dcc6fb90a7c32b18f883e10a3cb301901a428882d8d2015.pdf_2.jpg` | research_report | simplified_chinese | 1 | 0 | 11 |
| 121 | `page-2abc2a95-d39a-48db-9b6d-f40410412694.png` | research_report | traditional_chinese | 2 | 0 | 1 |
| 122 | `page-b9d507e7-1239-42de-b70a-d7ca65393dc9.png` | research_report | traditional_chinese | 2 | 0 | 4 |
| 123 | `page-c7792da7-4167-4f5c-a7ca-ec0f8833f83b.png` | research_report | traditional_chinese | 1 | 0 | 0 |
| 124 | `page-d0b2bb59-ce8e-4a1c-9ba1-a805606a477f.png` | research_report | traditional_chinese | 1 | 0 | 2 |
| 125 | `page-fd7003ba-9af2-4b60-8c45-1e2ac1c7f474.png` | research_report | traditional_chinese | 2 | 0 | 5 |

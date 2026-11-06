# AG-15 – Phân tích lỗi test

- Số mẫu test: 7600
- Tổng dự đoán sai: 665
- Mẫu sai được chọn: 20; đã ghi nhận nguyên nhân: 20
- Trạng thái: reviewed

## Các cặp nhầm lẫn trên toàn bộ test

| Nhãn thật | Nhãn dự đoán | Số mẫu sai |
|---|---|---|
| Sci/Tech | Business | 165 |
| Business | Sci/Tech | 151 |
| World | Business | 81 |
| World | Sci/Tech | 63 |
| Business | World | 52 |
| World | Sports | 50 |
| Sci/Tech | World | 49 |
| Sports | Business | 22 |
| Sports | World | 12 |
| Sci/Tech | Sports | 9 |
| Sports | Sci/Tech | 7 |
| Business | Sports | 4 |

## Nhóm nguyên nhân trong mẫu đã review

Các nhóm dưới đây được gán khi review và chỉ mô tả mẫu được chọn.

Codex đọc từng văn bản và đề xuất nhóm/giải thích. Đây là giả thuyết định tính, chưa phải nguyên nhân được xác minh bằng cơ chế model; nhóm cần review lại, nhất là nghi vấn nhãn nguồn. Mẫu lấy luân phiên theo cặp nhầm lẫn nên tỷ lệ nhóm không đại diện cho toàn bộ test. Không sửa nhãn thật, không dùng test để chọn lại cấu hình. token_count tính bằng tokenizer của checkpoint trước truncation, gồm special tokens; over_max_length chỉ ghi nhận độ dài, không khẳng định nguyên nhân lỗi.

| Nhóm nguyên nhân | Số mẫu |
|---|---|
| topic_overlap | 10 |
| label_issue | 4 |
| insufficient_context | 4 |
| entity_ambiguity | 2 |

## Mẫu được chọn

| Dòng CSV (tính từ 0, không tính header) | Nhãn thật | Dự đoán | Văn bản | Nhóm | Giải thích |
|---|---|---|---|---|---|
| 4058 | Sci/Tech | Business | Regulators Approve Artificial Heart The Food and Drug Administration approved the use of an artificial heart made by SynCardia Systems as a temporary device for people awaiting transplants. | entity_ambiguity | Bài nói về phê duyệt tim nhân tạo và cấy ghép, phù hợp Sci/Tech. Tên hãng SynCardia Systems có thể tạo liên tưởng doanh nghiệp; chưa có bằng chứng nội tại rằng đây là tín hiệu model dùng để chọn Business. |
| 5431 | Business | Sci/Tech | Vodafone begins 3G service Vodafone launches its third-generation services for mobile phones, offering video calls, music downloads and games. | topic_overlap | Vodafone ra mắt dịch vụ là hoạt động kinh doanh, nhưng mô tả tập trung 3G, điện thoại, gọi video và tải nhạc. Nội dung chứa cả tín hiệu Business và Sci/Tech. |
| 2306 | World | Business | Deal in Congress to Keep Tax Cuts, Widening Deficit Republican and Democratic leaders agreed to extend \$150 billion worth of tax cuts sought by President Bush without trying to pay for them. | topic_overlap | Quyết định của Quốc hội và Tổng thống là tin chính trị, trong khi cắt thuế 150 tỷ USD và thâm hụt là tín hiệu kinh tế rõ. Hai chủ đề cùng hiện diện trong đoạn trích. |
| 3157 | World | Sci/Tech | Gordon Cooper, NASA Mercury Pioneer, Dies LOS ANGELES - Gordon Cooper, who was the youngest and perhaps cockiest member of the original Mercury astronauts and set the space endurance record that helped clear the way for the first moon landing, has died. He was 77... | topic_overlap | Tin qua đời của Gordon Cooper có dạng tin thời sự World; NASA, phi hành gia Mercury và chuyến bay Mặt Trăng đồng thời tạo chủ đề khoa học vũ trụ. Chưa xác định được trọng số của từng tín hiệu trong model. |
| 5067 | Business | World | Quick end to US election crucial US President George W. Bush is on the verge of a re-election victory, but Democratic challenger John Kerry is not conceding defeat, at least not now. | label_issue | Đoạn trích chỉ nói bầu cử Bush/Kerry, không nêu tác động kinh tế. Dự đoán World phù hợp nội dung nhìn thấy; cần đối chiếu bài gốc và cách gán nhãn Business, giữ nguyên nhãn chuẩn hiện tại. |
| 3186 | World | Sports | Tennis: Night final for Aus Open The centenary Australian Open will be the first Grand Slam event to stage its final at night. | label_issue | Tiêu đề có Tennis và đoạn trích nói chung kết Australian Open/Grand Slam. Dự đoán Sports hợp lý về nội dung; nhãn nguồn World cần được nhóm kiểm tra, chưa tự kết luận hoặc sửa nhãn. |
| 2094 | Sci/Tech | World | The battle for DR Congo's wildlife As donors pledge \$40m to save DR Congo's wildlife, life in Virunga Park remains a battle ground. | topic_overlap | Bảo tồn động vật hoang dã thuộc chủ đề khoa học/môi trường, nhưng Congo, chiến sự và tài trợ quốc tế tạo ngữ cảnh World. Đoạn trích nhấn mạnh cả bảo tồn lẫn xung đột. |
| 7278 | Sports | Business | Report Says N.H.L. Will Reject Proposal The N.H.L. appears poised to reject a proposal made by the players union, which included a 24 percent reduction in pay and other concessions but not a hard salary cap. | topic_overlap | NHL và công đoàn cầu thủ xác định bối cảnh thể thao. Giảm lương 24%, nhượng bộ và trần lương là nội dung tài chính/lao động dễ trùng Business; đây là giả thuyết nội dung, không phải kết luận về cơ chế model. |
| 344 | Sports | World | Glory Comes Amid Empty Seats and Closed Shutters HERE in Old Europe, people install shutters outside their windows to keep out the heat, the pollution, the daylight, the noise. They also lock the shutters tight when they go away on holiday. | insufficient_context | Tiêu đề không nêu môn thể thao, đoạn trích chỉ nói cửa chớp và kỳ nghỉ ở châu Âu. Phần nhìn thấy thiếu ngữ cảnh giải thích nhãn Sports; chưa đủ bằng chứng xác định lý do model chọn World. |
| 5338 | Sci/Tech | Sports | #39;Halo 2 #39; has game fans ATLANTA - For many video game addicts, the buzz on the sequel to quot;Halo quot; is louder than a machine gun rat-a-tatting in their ears. | entity_ambiguity | Halo 2 là trò chơi điện tử trong chính đoạn trích. Các từ game và fans cũng dùng trong thể thao; việc phân biệt trò chơi điện tử với môn thể thao là giả thuyết hợp lý cho lỗi này, chưa được xác minh bằng giải thích model. |
| 1840 | Sports | Sci/Tech | Icing call Out of money, out of patience, out of time, and for the foreseeable future, out of business. | insufficient_context | Icing có thể gợi hockey, nhưng đoạn trích rất ngắn chỉ nói hết tiền, kiên nhẫn và thời gian, không nêu giải đấu hay đội. Không đủ ngữ cảnh để giải thích riêng dự đoán Sci/Tech. |
| 6185 | Business | Sports | Special to ESPN.com It #39;s the age old question: quot;What do you give to the man who #39;s been everything? quot;. Only time will tell whether Phil Knight #39;s retirement will be as long-lived as so many players he paid to endorse Nike. | topic_overlap | Phil Knight nghỉ hưu và Nike là thông tin doanh nghiệp, trong khi ESPN, cầu thủ và hợp đồng quảng bá gợi mạnh thể thao. Hai nhóm tín hiệu cùng hiện diện. |
| 7418 | Sci/Tech | Business | MicroStrategy Turnover Draws Analyst Scrutiny MicroStrategy Inc. said yesterday that president and chief financial officer Eric F. Brown had resigned and that founder Michael J. Saylor would again hold the company's top three jobs, prompting some analysts to raise concerns about the company's stock. | label_issue | Nội dung nhìn thấy nói thay đổi lãnh đạo, giám đốc tài chính và lo ngại cổ phiếu MicroStrategy, phù hợp Business. Nhãn Sci/Tech có thể theo chuyên mục doanh nghiệp công nghệ; cần đối chiếu nhãn gốc thay vì tự đổi chuẩn. |
| 5020 | Business | Sci/Tech | AOL Is Said to Plan 700 Layoffs America Online, the country #39;s leading Internet service, is preparing to lay off as many as 700 of its 13,000 employees in the United States, according to an executive knowledgeable about its plans. | topic_overlap | Sa thải 700 nhân viên là sự kiện kinh doanh/lao động. AOL và dịch vụ Internet đồng thời tạo ngữ cảnh công nghệ, có thể làm ranh giới Business/Sci/Tech khó phân biệt. |
| 3747 | World | Business | Ban for former Ahold executives Dutch retailer Ahold's former chairman and its ex-finance officer are barred from executive posts as part of a US fraud case settlement. | topic_overlap | Vụ dàn xếp gian lận tại Mỹ là sự kiện pháp lý/thời sự. Tập đoàn bán lẻ, cựu chủ tịch và lãnh đạo tài chính Ahold làm nội dung đồng thời mang chủ đề Business. |
| 511 | World | Sci/Tech | Mich. Elephant Gets Therapy for Arthritis ROYAL OAK, Mich. - Like any patient, Wanda needs positive reinforcement to wrestle through her physical therapy... | topic_overlap | Bài có dạng tin địa phương về một con voi, nhưng viêm khớp và vật lý trị liệu là tín hiệu y học/khoa học. Quy ước chuyên mục World so với Sci/Tech cần ngữ cảnh nguồn để phân định chắc chắn. |
| 7030 | Business | World | Without Batting An Eyelash : A very French welcome for a new <b>...</b> THAT night, it seemed as if three or four parties were going on at once in Cosmo Manille. Wrong, palanggas. Actually, there were five in Makati alone, and one of them had a stream of traffic | insufficient_context | Tiêu đề bị lược và đoạn trích chỉ kể tiệc tại Cosmo Manille/Makati, không nêu rõ kinh tế hay sự kiện quốc tế. Phần cung cấp thiếu ngữ cảnh để phân định nhãn Business; chưa quy lỗi cho truncation của tokenizer. |
| 2591 | World | Sports | Sports Court Hears Hamm Gold Medal Appeal LAUSANNE, Switzerland - Paul Hamm appeared before the sports world's highest court Monday to argue why he should he keep his Olympic gymnastics gold medal. The Court of Arbitration for Sport convened to hear the appeal from a South Korean gymnast who believes he was unfairly deprived of the gold in the men's all-around event in Athens last month because of a scoring error... | label_issue | Bài nói trực tiếp về huy chương Olympic, Paul Hamm và tranh chấp điểm thể dục dụng cụ. Dự đoán Sports có cơ sở; nhãn World có thể theo chuyên mục pháp lý/quốc tế, cần kiểm tra nguồn và giữ nguyên nhãn đánh giá. |
| 2423 | Sci/Tech | World | Internet Emerging As Potent Terrorist Tool By THOMAS WAGNER LONDON (AP) -- The images coming out of the latest hostage crisis in Iraq - capped by dramatic video of British captive Kenneth Bigley begging for his life - have transfixed Britons, left governments looking helpless, and revived a classic dilemma about whether to negotiate with terrorists. But the plight of the British construction worker and his two murdered American colleagues has also raised new concerns about terrorists' tremendous ability to set agendas in an Internet age that makes their messages - even in the form of shocking beheading videos - all but impossible to stop... | topic_overlap | Tiêu đề đặt trọng tâm Internet như công cụ khủng bố, còn phần thân tập trung con tin, Iraq và phản ứng chính phủ. Các tín hiệu công nghệ và World cùng xuất hiện; vượt giới hạn token nếu có không tự chứng minh nguyên nhân lỗi. |
| 1932 | Sports | Business | Woods, Mickelson form dynamic US duo For three days, it had been about dinners, galas, black-tie affairs, and enough social engagements to please Paris Hilton. | insufficient_context | Tiêu đề nêu Woods/Mickelson nhưng phần mô tả chủ yếu là tiệc và giao tiếp xã hội, không nhắc golf hay giải đấu. Thiếu ngữ cảnh thể thao có thể làm bài khó phân loại; chưa đủ chứng cứ về lý do chọn Business. |

Cần review ít nhất 20 mẫu sai thực tế để đạt tiêu chí AG-15. Nếu mô hình có ít hơn 20 lỗi, báo số thực có và giới hạn này; không bổ sung mẫu giả.

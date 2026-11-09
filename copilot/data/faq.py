"""Help-centre articles for the fictional Kanz marketplace (policies are invented)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Article:
    article_id: str
    topic: str
    question_ar: str
    answer_ar: str
    question_en: str
    answer_en: str

    def document(self) -> str:
        return " | ".join([self.question_ar, self.answer_ar, self.question_en, self.answer_en])


ARTICLES: list[Article] = [
    Article(
        "FAQ-01",
        "redeem",
        "إزاي أستخدم الكوبون عند التاجر؟",
        "افتح الطلب من حسابك، واعرض كود الكوبون (QR) للتاجر قبل الدفع. التاجر بيعمل مسح للكود وبيتفعل الخصم فورًا.",
        "How do I redeem a voucher at the merchant?",
        "Open the order in your account and show the voucher QR code to the merchant before paying; they scan it and the discount applies immediately.",
    ),
    Article(
        "FAQ-02",
        "refund",
        "ينفع أرجّع الكوبون وآخد فلوسي؟",
        "أيوه، تقدر تطلب استرداد خلال ١٤ يوم من الشراء طالما الكوبون ما اتستخدمش. الفلوس بترجع بنفس طريقة الدفع خلال ٥ إلى ٧ أيام عمل.",
        "Can I return a voucher and get a refund?",
        "Yes. You can request a refund within 14 days of purchase as long as the voucher is unused. Money returns to the original payment method within 5-7 business days.",
    ),
    Article(
        "FAQ-03",
        "refund",
        "الكوبون خلص ومستخدمتوش، هل ينفع استرداد؟",
        "الكوبونات المنتهية مش بتترد فلوس، لكن بنحوّل قيمتها لرصيد في محفظة كنز صالح ٩٠ يوم.",
        "My voucher expired unused. Can I get a refund?",
        "Expired vouchers are not refunded in cash, but their value is converted to Kanz wallet credit valid for 90 days.",
    ),
    Article(
        "FAQ-04",
        "payment",
        "إيه طرق الدفع المتاحة؟",
        "تقدر تدفع بالكارت (فيزا أو ماستركارد)، أو المحفظة الإلكترونية، أو رصيد محفظة كنز. الدفع كاش عند التاجر مش متاح للكوبونات.",
        "Which payment methods are available?",
        "Cards (Visa or Mastercard), mobile wallets, or Kanz wallet credit. Cash at the merchant is not available for vouchers.",
    ),
    Article(
        "FAQ-05",
        "payment",
        "اتخصم مني فلوس والطلب ما ظهرش",
        "ساعات الخصم بيتعلق لحد ٢٤ ساعة. لو الطلب ما ظهرش بعد كده، البنك بيرجع المبلغ تلقائي. لو عدّى ٤٨ ساعة كلمنا برقم العملية.",
        "I was charged but the order did not appear",
        "A pending charge can take up to 24 hours. If no order appears, your bank reverses it automatically. After 48 hours, contact us with the transaction reference.",
    ),
    Article(
        "FAQ-06",
        "account",
        "نسيت الباسورد، أعمل إيه؟",
        'دوس على "نسيت كلمة السر" في صفحة الدخول، وهيوصلك كود على موبايلك أو إيميلك تغيّر بيه كلمة السر.',
        "I forgot my password. What do I do?",
        'Tap "Forgot password" on the sign-in page; we send a code to your phone or email to set a new one.',
    ),
    Article(
        "FAQ-07",
        "account",
        "إزاي أغيّر رقم الموبايل في الحساب؟",
        "من الإعدادات ← البيانات الشخصية ← رقم الموبايل. هنبعتلك كود تأكيد على الرقم الجديد.",
        "How do I change my phone number?",
        "Go to Settings → Personal details → Phone number. We send a confirmation code to the new number.",
    ),
    Article(
        "FAQ-08",
        "merchant",
        "التاجر رفض الكوبون، أعمل إيه؟",
        "اطلب من التاجر يتأكد إن الكوبون لسه صالح وفي نفس الفرع. لو لسه رافض، افتح بلاغ من صفحة الطلب وهنرجعلك الفلوس كاملة لو المشكلة من التاجر.",
        "The merchant refused my voucher. What should I do?",
        "Ask the merchant to check the voucher is valid for that branch. If they still refuse, open a report from the order page and we refund you in full if the merchant is at fault.",
    ),
    Article(
        "FAQ-09",
        "merchant",
        "ينفع أستخدم الكوبون في أي فرع؟",
        'الكوبون صالح في الفرع المكتوب في تفاصيل العرض بس، إلا لو مكتوب "كل الفروع".',
        "Can I use the voucher at any branch?",
        'Only at the branch listed in the offer details, unless the offer says "all branches".',
    ),
    Article(
        "FAQ-10",
        "booking",
        "لازم أحجز قبل ما أروح؟",
        "عروض السبا والعيادات والفنادق محتاجة حجز مسبق مع التاجر. باقي العروض مش محتاجة حجز إلا لو مكتوب في الشروط.",
        "Do I need to book before going?",
        "Spa, clinic and hotel offers require booking with the merchant in advance. Other offers do not, unless the terms say so.",
    ),
    Article(
        "FAQ-11",
        "booking",
        "إزاي ألغي حجز الفندق؟",
        "الإلغاء مجاني لحد ٤٨ ساعة قبل الوصول من صفحة الطلب. بعد كده الإلغاء بيتحسب عليه ليلة.",
        "How do I cancel a hotel booking?",
        "Cancellation is free up to 48 hours before arrival from the order page. After that, one night is charged.",
    ),
    Article(
        "FAQ-12",
        "wallet",
        "رصيد محفظة كنز بيخلص إمتى؟",
        "رصيد الاسترداد صالح ٩٠ يوم، ورصيد العروض الترويجية صالح ٣٠ يوم. تقدر تشوف تاريخ الانتهاء في صفحة المحفظة.",
        "When does Kanz wallet credit expire?",
        "Refund credit is valid for 90 days and promotional credit for 30 days. The expiry date is shown on the wallet page.",
    ),
    Article(
        "FAQ-13",
        "gift",
        "ينفع أهدي كوبون لحد؟",
        'أيوه، من صفحة العرض اختار "إهداء" واكتب رقم موبايل الشخص، هيوصله الكوبون في رسالة.',
        "Can I gift a voucher to someone?",
        'Yes. On the offer page choose "Gift" and enter their phone number; they receive the voucher by message.',
    ),
    Article(
        "FAQ-14",
        "order",
        "فين ألاقي الكوبون بعد ما اشتريت؟",
        'الكوبون بيظهر في "طلباتي" وبيوصلك على الإيميل كمان. لو مش لاقيه جرّب تعمل تحديث للتطبيق.',
        "Where do I find my voucher after buying?",
        'It appears under "My orders" and is also emailed to you. If it is missing, refresh the app.',
    ),
    Article(
        "FAQ-15",
        "order",
        "أعرف حالة طلبي إزاي؟",
        'من "طلباتي" هتلاقي حالة كل طلب: مدفوع، مستخدم، مسترد، أو منتهي. وتقدر تسأل المساعد برقم الطلب.',
        "How can I check my order status?",
        '"My orders" shows each order\'s status: paid, redeemed, refunded or expired. You can also ask the assistant with your order number.',
    ),
    Article(
        "FAQ-16",
        "privacy",
        "بياناتي الشخصية بتستخدم في إيه؟",
        "بنستخدم بياناتك علشان نكمل الطلب ونتواصل معاك بس. ما بنبيعش بياناتك، وتقدر تطلب مسح حسابك من الإعدادات.",
        "How is my personal data used?",
        "Only to fulfil orders and contact you. We never sell your data, and you can request account deletion from Settings.",
    ),
    Article(
        "FAQ-17",
        "support",
        "إزاي أكلم خدمة العملاء؟",
        "من التطبيق: المساعدة ← تواصل معانا، وخدمة العملاء متاحة من ٩ الصبح لـ ١١ بالليل كل يوم.",
        "How do I contact customer service?",
        "In the app: Help → Contact us. Customer service is available 9 am to 11 pm every day.",
    ),
    Article(
        "FAQ-18",
        "promo",
        "عندي كود خصم بس مش شغال",
        "اتأكد إن الكود لسه صالح وإن قيمة الطلب فوق الحد الأدنى المكتوب في شروط الكود. الكود بيتستخدم مرة واحدة لكل حساب.",
        "My promo code is not working",
        "Check the code has not expired and the order is above the code's minimum value. Each code can be used once per account.",
    ),
]

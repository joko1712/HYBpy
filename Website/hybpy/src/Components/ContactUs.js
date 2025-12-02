
export const handleContactUsClick = () => {
  const email1Encoded = "ai5wZWRyZWlyYUBjYW1wdXMuZmN0LnVubC5wdA==";
  const email2Encoded = "cnMuY29zdGFAZmN0LnVubC5wdA==";

  const email1 = atob(email1Encoded);
  const email2 = atob(email2Encoded);

  const mailtoLink = `mailto:${email1},${email2}`;
  window.location.href = mailtoLink;
};
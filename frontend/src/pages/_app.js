import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return (
    <>
      {/* 공통 레이아웃 */}
      <header></header>
      <Component {...pageProps} />
      <footer></footer>
    </>
  );
}

export default MyApp;

import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return (
    <>
      {/* 공통 레이아웃 */}
      <header>Header</header>
      <Component {...pageProps} />
      <footer>Footer</footer>
    </>
  );
}

export default MyApp;

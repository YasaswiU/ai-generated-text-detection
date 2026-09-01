import { useState } from "react";
import Header from "./components/Header.jsx";
import Footer from "./components/Footer.jsx";
import Home from "./pages/Home.jsx";
import About from "./pages/About.jsx";

export default function App() {
  const [page, setPage] = useState("home");

  return (
    <>
      <Header />
      <nav className="site-nav">
        <button
          className={`site-nav__link${page === "home" ? " site-nav__link--active" : ""}`}
          onClick={() => setPage("home")}
        >
          Analyze
        </button>
        <button
          className={`site-nav__link${page === "about" ? " site-nav__link--active" : ""}`}
          onClick={() => setPage("about")}
        >
          About
        </button>
      </nav>
      {page === "home" ? <Home /> : <About />}
      <Footer />
    </>
  );
}

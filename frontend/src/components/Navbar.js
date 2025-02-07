import { useRouter } from 'next/router';
import React, { useState, useEffect } from 'react';
import Info from '../components/Info';

export default function Navbar({ }) {
  const router = useRouter();

  const goAttack = () => {
    router.push('/attack');
  };

  const goMain = () => {
    router.push('/');
  };

  const [showInfo, setShowInfo] = useState(false);

  return (
    <nav className="w-20px bg-gray-200 p-4 rounded-l-lg flex flex-col items-center">
      {/* Navigation Buttons */}
      <button
        className="w-12 h-12 flex items-center justify-center mb-4 bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400"
        onClick={goMain}
      >
        <span className="material-icons">face</span>
      </button>
      <button
        className="w-12 h-12 flex items-center justify-center mb-4 bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400"
        onClick={goAttack}
      >
        <span className="material-icons">face_retouching_off</span>
      </button>
      <button
        className="w-12 h-12 flex items-center justify-center bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400"
        onClick={() => setShowInfo(true)}
      >
        <span className="material-icons">info</span>
      </button>
      <Info
        show={showInfo}
        onClose={() => setShowInfo(false)}
      />
    </nav>
  );
}
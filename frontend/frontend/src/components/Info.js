import React, { useState, useEffect } from 'react';
import Image from 'next/image';

export default function INFO({ show, onClose }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-30 z-50">
      <div className="relative bg-white rounded-lg shadow-lg p-6 w-96">
        <h1 className="text-2xl font-bold mb-4 text-center text-black">프로메테우스 4팀 Deep Safe</h1>
        <p className="mb-2 text-gray-700">
          팀장 : 오유진
        </p>
        <p className="mb-4 text-gray-700">
          부원 : 강민수, 이다솔, 조현우
        </p>
        <div className="flex justify-center">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-400 text-white rounded-md hover:bg-blue-500 transition"
          >닫기</button>
        </div>
      </div>
    </div>
  );
}

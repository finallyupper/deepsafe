'use client';
import React, { useState, useEffect } from 'react';


export default function INFO({ show, onClose }) {
  if (!show) return null;
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-30 z-50">
      <div className="relative flex flex-col justify-center items-center w-[32vw]">
        <img
          src="/panel.png"
          alt="Public Image"
          className="object-fit"
        />
        <div className="flex justify-center">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-400 text-white rounded-md shadow-lg font-semibold mt-6 hover:bg-blue-500 transition"
          >닫기</button>
        </div>
      </div>
    </div>
  );
}

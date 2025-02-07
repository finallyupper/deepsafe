import { useState } from "react";
import { useRouter } from 'next/router';

import Navbar from "../components/Navbar";

export default function Home() {
  const router = useRouter();


  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-full h-full  p-3 ">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100 rounded-lg">
          {/* Navigation Bar */}
          <Navbar  />

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-blue-500">
              Deep Safe
            </h1>
            <p className="mt-4 text-gray-600">
              Protect face swaps. We will watermark the image if you upload some image.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
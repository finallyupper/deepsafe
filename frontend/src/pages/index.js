import { useState } from "react";
import { useRouter } from 'next/router';
import Info from '../components/Info';

export default function Home() {
  const router = useRouter();

  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [watermarkedImage, setWatermarkedImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  const goAttack = () => {
    router.push('/attack');
  };

  // 파일 업로드 핸들러
  const handleImageUpload = (e) => {
    const file = e.target.files[0]; // 업로드된 파일 가져오기
    if (file) {
      setImage(file); // 파일 상태 저장
      setImagePreview(URL.createObjectURL(file)); // 미리보기 URL 생성
    }
  };
  
  // 워터마킹 요청 핸들러
  const watermark = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", image); // 서버로 전송할 이미지 추가

      // API 요청
      const response = await fetch("https://api-endpoint.com/watermark", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process the image.");
      }

      const data = await response.json(); // 서버로부터 데이터 받기
      setWatermarkedImage(data.watermarkedImageUrl); // 워터마킹된 이미지 URL 저장
    } catch (error) {
      console.error("Error during watermarking:", error);
      alert("Failed to watermark the image. Please try again.");
    } finally {
      setIsLoading(false); // 로딩 종료
    }
  };

  // 워터마킹된 이미지 다운로드 핸들러
  const downloadImage = () => {
    const link = document.createElement("a");
    link.href = watermarkedImage; // 워터마킹된 이미지 URL
    link.download = "watermarked_image.png"; // 다운로드될 파일 이름
    link.click(); // 클릭 이벤트로 다운로드 시작
  };

  // 로딩 스피너 컴포넌트
  const LoadingSpinner = () => {
    return (
      <div className="flex items-center justify-center mt-2">
        <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-3/4 h-3/4 bg-gradient-to-r from-teal-400 via-pink-500 to-yellow-500 p-1 rounded-lg">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100 rounded-lg">
          {/* Navigation Bar */}
          <nav className="w-1/5 bg-gray-200 p-4 rounded-l-lg flex flex-col items-center">

            {/* Additional Navigation Items */}
            <button className="w-12 h-12 flex items-center justify-center mb-4 bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400">
              <span className="material-icons">face</span>
            </button>
            <button
              className="w-12 h-12 flex items-center justify-center mb-4 bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400"
              onClick={goAttack}
            ><span class="material-icons">face_retouching_off</span>
            </button>
            <button
              className="w-12 h-12 flex items-center justify-center bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400"
              onClick={() => setShowInfo(true)}
            ><span className="material-icons">info</span></button>
          </nav>

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-blue-500">
              Deep Safe
            </h1>
            <p className="mt-4 text-gray-600">
              Protect face swaps. We will watermark the image if you upload some image.
            </p>
            <button
              className='bg-blue-400 text-white p-3 rounded shadow text-2xl mt-2 hover:bg-blue-500'
              onClick={() => document.getElementById("image-upload").click()}
            >Upload image</button>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageUpload}
            />
            {/* Images */}
            <div className="flex mt-6">
              {/* Image Preview */}
              {imagePreview && (
                <div className="flex flex-col">
                  <h2 className="text-xl font-semibold text-gray-800">Image:</h2>
                  <img
                    src={imagePreview}
                    alt="Uploaded Preview"
                    className="mt-4 max-w-full h-auto"
                  />
                  <button
                    className="mt-2 rounded p-1 text-white bg-blue-400 text-2xl"
                    onClick={watermark}
                  >{isLoading ? "Processing..." : "Watermark"}</button>
                  {isLoading && <LoadingSpinner />}
                </div>
              )}
              
              {/* Watermarked Image Preview */}
              {watermarkedImage && (
                <div className="">
                  <h2 className="text-xl font-semibold text-gray-800">Watermarked Image:</h2>
                  <img
                    src={watermarkedImage}
                    alt="Watermarked Preview"
                    className="mt-4 max-w-full h-auto"
                  />
                  <button
                    className="mt-2 rounded p-1 text-white bg-blue-400 text-2xl"
                    onClick={downloadImage}
                  >Download</button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <Info
        show={showInfo}
        onClose={() => setShowInfo(false)}
      />
    </div>
  );
}

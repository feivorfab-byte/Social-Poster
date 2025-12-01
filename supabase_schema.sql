-- =============================================
-- STUDIO LIGHTS DATABASE SCHEMA
-- =============================================

-- Prompts table: Store and version prompt templates
CREATE TABLE prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Lighting schemes: Configurable lighting options
CREATE TABLE lighting_schemes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    prompt_text TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Default backgrounds: White, Gray, Black descriptions
CREATE TABLE backgrounds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    is_default BOOLEAN DEFAULT false,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Generation logs: Track usage and performance (optional but useful)
CREATE TABLE generation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    orientation TEXT,
    lighting_scheme TEXT,
    background_type TEXT,
    quality TEXT,
    has_master BOOLEAN DEFAULT false,
    has_cached_bg BOOLEAN DEFAULT false,
    verification_passed BOOLEAN,
    verification_attempts INTEGER DEFAULT 1,
    generation_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================
-- INSERT DEFAULT DATA
-- =============================================

-- Default prompts
INSERT INTO prompts (name, content) VALUES
('analysis_metadata', 'Analyze this product photograph and extract metadata.

ORIENTATION: Is the product lying flat (viewed from above, like clothing on a table) or standing upright (viewed at eye level, like a bottle or statue)?
- "flat_lay" = lying flat, top-down view
- "standing" = upright, eye-level view  
- "angled" = neither clearly flat nor standing

CAMERA ANGLE: Describe the camera perspective in 3-5 words (e.g., "overhead flat lay", "eye-level front view", "3/4 elevated view")

PRODUCT DIMENSIONS: Estimate real-world size as "W x H x D" with units (inches or feet). Example: "12 x 8 x 4 inches"

VISIBLE TEXT: List any text, numbers, logos, or brand names visible on the product. Transcribe exactly. Empty string if none.

JSON only:
{
    "orientation": "flat_lay" or "standing" or "angled",
    "camera_angle": "brief description",
    "product_dimensions": "W x H x D units",
    "visible_text": "exact text or empty"
}'),

('composition_flat_lay', 'COMPOSITION: Top-down flat lay photograph. Camera positioned directly above, looking straight down. Product lies flat on a horizontal surface. Center the product, filling 50-60% of frame. Background extends as continuous horizontal plane around all edges. Soft contact shadow directly beneath product.'),

('composition_standing', 'COMPOSITION: Standing product photograph. Camera at eye level or slightly elevated. Product stands upright on surface with depth perspective - background visible beneath and behind, receding naturally. Center product, filling 50-60% of frame height. Natural contact shadow at base.'),

('composition_angled', 'COMPOSITION: Photograph at natural angle matching reference. Center product, filling 50-60% of frame. Background surface visible around product with appropriate perspective. Soft contact shadow grounding product on surface.'),

('output_quality', 'OUTPUT: Authentic studio photograph. Natural depth of field, real material textures, unified lighting across product and background. Shot on full-frame camera with 90mm lens at f/8.'),

('verification', 'Compare these two images. Image 1 is the original product. Image 2 is a generated studio photograph.

Verify:
1. PRODUCT FIDELITY: Same product? Same shape, proportions, colors, materials, details?
2. ORIENTATION: Should be "{orientation}" (flat_lay=top-down, standing=eye-level with depth, angled=natural angle). Correct?
3. COMPOSITION: Product centered, filling ~50-60% of frame?
4. LIGHTING: Unified across product and background?
{text_check}

JSON:
{"product_ok": bool, "orientation_ok": bool, "composition_ok": bool, "lighting_ok": bool, {text_field}"pass": bool, "issues": []}'),

('background_reproduction', 'Reproduce this image exactly as a clean studio photography surface.

IMAGE 1 shows a background/surface material. Create an exact copy preserving:
- All colors, textures, patterns exactly
- ALL text, writing, numbers, logos - exact content, placement, size, style
- ALL graphics, drawings, marks exactly

Fill entire frame, evenly lit. Accuracy is critical.');

-- Default lighting schemes
INSERT INTO lighting_schemes (id, name, description, prompt_text, sort_order) VALUES
('highkey', 'High Key', 'Bright, airy lighting with minimal shadows. Clean and modern.',
'LIGHTING: High-key. Multiple soft sources from all directions, primary from above. Shadows minimal at 10-20% gray. Highlights soft and diffused. Exposure bright (+0.5 stop). Background evenly lit, matching or brighter than product. Color temperature neutral-cool (5500-6000K).', 1),

('softbox', 'Soft Box', 'Even, diffused lighting that minimizes harsh shadows. Classic commercial look.',
'LIGHTING: Soft box. Large diffused source at 45° left and above, subtle fill from right. Shadows soft gradients at 30-40% gray with smooth falloff. Highlights broad and wrapped. Exposure balanced and neutral. Background evenly lit matching product. Color temperature neutral daylight (5500K).', 2),

('product', 'Product Standard', 'Multi-light setup for complete visibility. Shows all details clearly.',
'LIGHTING: Product standard. Main light 45° left-above, fill from right, edge light from behind. Shadows light at 25-35% gray - all details visible. Highlights clean and controlled. Exposure accurate and color-correct. Background evenly lit. Color temperature precise neutral daylight.', 3),

('butterfly', 'Butterfly', 'Top-down lighting creating symmetrical shadows. Elegant and glamorous.',
'LIGHTING: Butterfly/Paramount. Primary source directly above and slightly forward, angling down. Shadows symmetrical, falling straight down at 35-45% gray. Highlights centered on top surfaces. Background slightly darker than product. Color temperature neutral to warm (5000-5500K).', 4),

('natural', 'Natural Window', 'Simulates soft window light. Warm, organic, and inviting.',
'LIGHTING: Natural window. Soft directional light from camera-left, ambient fill on shadow side. Shadows organic at 40-50% gray with gradual falloff. Highlights soft sheens. Background shows gentle gradient from lit to shadow side. Color temperature warm daylight (5000-5200K) with subtle golden tones.', 5),

('gradient', 'Gradient', 'Smooth light-to-dark transition across background. Subtle and sophisticated.',
'LIGHTING: Gradient background. Product lit with standard 45° front-side light, shadows at 30-40% gray. Background has smooth continuous gradient from light on one side to dark on opposite side - no banding. Product properly exposed against graduated backdrop. Color temperature neutral.', 6),

('rembrandt', 'Rembrandt', 'Classic artistic lighting with triangle shadow. Adds depth and dimension.',
'LIGHTING: Rembrandt. Key light at 45° left and 45° above. Shadows rich at 50-60% gray with clear light/shadow separation. Signature: small triangle of light on shadow side. Highlights defined but not harsh. Background graduates lighter on key side to darker on shadow side. Color temperature warm (4500-5000K).', 7),

('rim', 'Rim/Edge', 'Backlight that creates a glowing outline. Makes products pop from background.',
'LIGHTING: Rim/edge. Primary light from behind product, subtle fill from front. Product front shows 40-50% gray shadows. Signature: bright glowing outline tracing all edges with slight overexposure, creating halo separation. Background darker than rim-lit edges. Rim light slightly cool, front fill neutral.', 8),

('spotlight', 'Spotlight', 'Focused beam creates dramatic pool of light. Theatrical and attention-grabbing.',
'LIGHTING: Spotlight. Focused concentrated beam from 45° left-above creates visible oval pool of light. Inside pool: shadows at 40-50% gray with bright hotspot. Outside pool: dark at 70-85% gray. Background shows spotlight pattern - bright where lit, dark elsewhere. Color temperature neutral to warm tungsten.', 9),

('lowkey', 'Low Key', 'Dramatic lighting with deep shadows. Creates mood and emphasizes texture.',
'LIGHTING: Low-key dramatic. Single hard source from 60-90° side at product height. Shadows deep at 85-95% darkness, dominating the image. Rapid falloff from light to dark. Highlights crisp with hard edges emphasizing texture. Contrast ratio 8:1 or higher. Background falls to deep shadow. Color temperature neutral to warm.', 10);

-- Default backgrounds
INSERT INTO backgrounds (id, name, description, is_default, sort_order) VALUES
('white', 'White', 'Professional seamless studio surface: pure white duvetyn fabric - the industry-standard material for photography. Duvetyn has a soft, velvety matte surface that completely absorbs light with zero shine or reflections. The fabric is taut and smooth with virtually no wrinkles or creases. The surface extends seamlessly with no visible edges or seams.', true, 1),

('gray', 'Gray', 'Professional seamless studio surface: neutral medium gray duvetyn fabric - the industry-standard material for photography. Duvetyn has a soft, velvety matte surface that completely absorbs light with zero shine or reflections. The gray is a true neutral (no warm or cool cast). The fabric is taut and smooth with virtually no wrinkles or creases. The surface extends seamlessly with no visible edges or seams.', true, 2),

('black', 'Black', 'Professional seamless studio surface: near-black duvetyn fabric - the industry-standard material for photography. Duvetyn has a soft, velvety matte surface that completely absorbs light with zero shine or reflections. Deep, rich black with just enough tone to show subtle fabric texture and prevent pure digital black. The fabric is taut and smooth with virtually no wrinkles or creases. The surface extends seamlessly with no visible edges or seams.', true, 3);

-- Create indexes for performance
CREATE INDEX idx_prompts_name ON prompts(name) WHERE is_active = true;
CREATE INDEX idx_lighting_schemes_active ON lighting_schemes(sort_order) WHERE is_active = true;
CREATE INDEX idx_generation_logs_created ON generation_logs(created_at DESC);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE prompts ENABLE ROW LEVEL SECURITY;
ALTER TABLE lighting_schemes ENABLE ROW LEVEL SECURITY;
ALTER TABLE backgrounds ENABLE ROW LEVEL SECURITY;
ALTER TABLE generation_logs ENABLE ROW LEVEL SECURITY;

-- Allow public read access (for the backend)
CREATE POLICY "Allow public read" ON prompts FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON lighting_schemes FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON backgrounds FOR SELECT USING (true);
CREATE POLICY "Allow public insert" ON generation_logs FOR INSERT WITH CHECK (true);

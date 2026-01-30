"""
Create Supabase tables for CourtMind
"""
import psycopg2

# Supabase connection details
DB_HOST = "db.ffdcfuckuzdejdmmaiyx.supabase.co"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "Jg6u&8yJWirc8Bp"

SQL_COMMANDS = """
-- Create profiles table for user subscription data
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT,
  subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'premium')),
  stripe_customer_id TEXT,
  stripe_subscription_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create tracked_picks table for bet history
CREATE TABLE IF NOT EXISTS public.tracked_picks (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
  player_name TEXT NOT NULL,
  team TEXT,
  opponent TEXT,
  stat_type TEXT NOT NULL,
  line DECIMAL(5,1) NOT NULL,
  direction TEXT CHECK (direction IN ('OVER', 'UNDER')) NOT NULL,
  projection DECIMAL(5,1),
  confidence INTEGER,
  dk_line DECIMAL(5,1),
  fd_line DECIMAL(5,1),
  edge DECIMAL(5,1),
  game_date DATE NOT NULL,
  result DECIMAL(5,1),
  hit BOOLEAN,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracked_picks ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (ignore errors)
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can view own picks" ON public.tracked_picks;
DROP POLICY IF EXISTS "Users can insert own picks" ON public.tracked_picks;
DROP POLICY IF EXISTS "Users can update own picks" ON public.tracked_picks;
DROP POLICY IF EXISTS "Users can delete own picks" ON public.tracked_picks;

-- RLS policy: users can only see their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = id);

-- RLS policy: users can only see their own picks
CREATE POLICY "Users can view own picks" ON public.tracked_picks
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own picks" ON public.tracked_picks
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own picks" ON public.tracked_picks
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own picks" ON public.tracked_picks
  FOR DELETE USING (auth.uid() = user_id);

-- Function to auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email)
  VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop trigger if exists
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Trigger to create profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
"""

def main():
    print("Connecting to Supabase database...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        conn.autocommit = True
        cursor = conn.cursor()

        print("Connected! Running SQL commands...")

        # Split and execute each command
        commands = [cmd.strip() for cmd in SQL_COMMANDS.split(';') if cmd.strip()]

        for i, cmd in enumerate(commands):
            if cmd:
                try:
                    cursor.execute(cmd + ';')
                    print(f"  [{i+1}/{len(commands)}] Success")
                except Exception as e:
                    print(f"  [{i+1}/{len(commands)}] Error: {e}")

        cursor.close()
        conn.close()
        print("\nDone! Tables created successfully.")

    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    main()

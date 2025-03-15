/**
 * Searches for news articles using multiple methods
 */
async function searchNews(query) {
  try {
    // Try using duckduckgo-search package if available
    try {
      console.log(`Searching for "${query}" using DuckDuckGo Search package...`);
      
      // Import the package correctly - duckduckgo-search has been changing its API
      const duckduckgo = require('duckduckgo-search');
      console.log('Available in duckduckgo-search:', Object.keys(duckduckgo));
      
      if (typeof duckduckgo.search === 'function') {
        // If it has a search function, use it directly
        console.log('Using duckduckgo.search function');
        const results = await duckduckgo.search(query + ' news', {
          time: 'd', // Last day
          max_results: 10
        });
        
        if (results && results.length > 0) {
          console.log(`Found ${results.length} results using duckduckgo.search`);
          return results.map(result => ({
            title: result.title,
            url: result.url || result.href,
            snippet: result.description || result.body || ''
          }));
        }
      } else if (duckduckgo.DDGS) {
        // If it has DDGS class
        console.log('Using DDGS class');
        const ddgs = new duckduckgo.DDGS();
        const results = await ddgs.news(query, {
          region: 'wt-wt',
          safe: 'off',
          time: 'd',
          max_results: 10
        });
        
        if (results && results.length > 0) {
          console.log(`Found ${results.length} results using DDGS class`);
          return results.map(result => ({
            title: result.title,
            url: result.url || result.href,
            snippet: result.body || result.description || ''
          }));
        }
      } else {
        console.log('No compatible search function found in duckduckgo-search package');
      }
    } catch (packageError) {
      console.error('DuckDuckGo Search package error:', packageError);
    }
    
    // Direct HTTP request to DuckDuckGo News
    try {
      console.log(`Trying direct HTTP request to DuckDuckGo News for "${query}"...`);
      const encodedQuery = encodeURIComponent(query + ' news');
      
      // Use more targeted URL to get news results directly
      const response = await axios.get(`https://duckduckgo.com/html/?q=${encodedQuery}&iar=news&ia=news`, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml',
          'Accept-Language': 'en-US,en;q=0.9',
          'Cache-Control': 'no-cache'
        },
        timeout: 15000
      });
      
      console.log('DuckDuckGo HTML response received, length:', response.data.length);
      
      const $ = cheerio.load(response.data);
      const results = [];
      
      // Try multiple selector patterns that might match news results
      $('.result, .result__body, .web-result').each((i, element) => {
        const titleElement = $(element).find('.result__title a, .result__a, .result-title a');
        const title = titleElement.text().trim();
        const url = titleElement.attr('href');
        
        const snippetElement = $(element).find('.result__snippet, .result__body, .result-snippet');
        const snippet = snippetElement.text().trim();
        
        if (title && url && snippet) {
          results.push({
            title,
            url: url.startsWith('/') ? `https://duckduckgo.com${url}` : url,
            snippet
          });
        }
      });
      
      if (results.length > 0) {
        console.log(`Found ${results.length} results from direct DuckDuckGo HTML request`);
        return results.slice(0, 10);
      } else {
        console.log('No results found in DuckDuckGo HTML response');
      }
    } catch (ddgHtmlError) {
      console.error('DuckDuckGo HTML request error:', ddgHtmlError.message);
    }
    
    // Try Bing News as an alternative
    try {
      console.log(`Trying Bing News for "${query}"...`);
      const encodedQuery = encodeURIComponent(query + ' news');
      
      const bingResponse = await axios.get(`https://www.bing.com/news/search?q=${encodedQuery}&qft=interval%3D%227%22`, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml',
          'Accept-Language': 'en-US,en;q=0.9'
        },
        timeout: 15000
      });
      
      const $bing = cheerio.load(bingResponse.data);
      const results = [];
      
      $bing('.news-card').each((i, element) => {
        const titleElement = $bing(element).find('a.title');
        const title = titleElement.text().trim();
        const url = titleElement.attr('href');
        
        const snippetElement = $bing(element).find('.snippet');
        const snippet = snippetElement.text().trim();
        
        if (title && url && snippet) {
          results.push({
            title,
            url,
            snippet
          });
        }
      });
      
      if (results.length > 0) {
        console.log(`Found ${results.length} results from Bing News`);
        return results.slice(0, 10);
      }
    } catch (bingError) {
      console.error('Bing News error:', bingError.message);
    }
    
    // Final fallback to Google News
    try {
      console.log(`Final fallback to Google News for "${query}"...`);
      const encodedQuery = encodeURIComponent(query + ' news');
      
      const googleResponse = await axios.get(`https://news.google.com/search?q=${encodedQuery}`, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml',
          'Accept-Language': 'en-US,en;q=0.9'
        },
        timeout: 15000
      });
      
      const $google = cheerio.load(googleResponse.data);
      const results = [];
      
      $google('article').each((i, element) => {
        const titleElement = $google(element).find('h3 a, h4 a');
        const title = titleElement.text().trim();
        
        // Google News uses relative URLs
        const relativeUrl = titleElement.attr('href');
        const url = relativeUrl ? 
          (relativeUrl.startsWith('/') ? 
            `https://news.google.com${relativeUrl}` : relativeUrl) : '';
            
        const snippetElement = $google(element).find('p, span[data-n-tid="9"]');
        const snippet = snippetElement.text().trim();
        
        if (title && url) {
          results.push({
            title,
            url,
            snippet: snippet || title
          });
        }
      });
      
      if (results.length > 0) {
        console.log(`Found ${results.length} results from Google News`);
        return results.slice(0, 10);
      } else {
        console.log('No results found in Google News');
      }
    } catch (googleError) {
      console.error('Google News error:', googleError.message);
    }
    
    console.log('All search methods failed. Returning empty results.');
    return [];
  } catch (error) {
    console.error('Critical error in search function:', error);
    return [];
  }
}// Breaking News Search Bot
// Searches for fresh news stories using DuckDuckGo and filters them based on criteria

// Required packages
// npm install dotenv telegraf axios openai moment cheerio duckduckgo-search

require('dotenv').config();
const { Telegraf } = require('telegraf');
const { DuckDuckGoSearch } = require('duckduckgo-search');
const { OpenAI } = require('openai');
const axios = require('axios');
const cheerio = require('cheerio');
const moment = require('moment');

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize Telegram bot
const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);

// Set time threshold for fresh news (8 hours)
const TIME_THRESHOLD = 8 * 60 * 60 * 1000; // 8 hours in milliseconds

// Welcome message
bot.start((ctx) => {
  ctx.reply('Welcome to Breaking News Bot! Send /search followed by a topic to find fresh news stories.');
});

// Help command
bot.help((ctx) => {
  ctx.reply(
    'How to use this bot:\n\n' +
    '/search [topic] - Search for breaking news on a specific topic\n' +
    'Example: /search earthquake california'
  );
});

// Main search command
bot.command('search', async (ctx) => {
  const query = ctx.message.text.replace('/search', '').trim();
  
  if (!query) {
    return ctx.reply('Please provide a search term. Example: /search earthquake california');
  }
  
  ctx.reply(`🔍 Searching for breaking news about: "${query}"`);
  
  try {
    // Generate expanded search queries using OpenAI
    const searchQueries = await generateSearchQueries(query);
    
    // Display the generated queries
    ctx.reply(`🧠 Generated search queries:\n${searchQueries.join('\n')}`);
    
    // Search for news using each query
    let allResults = [];
    
    for (const searchQuery of searchQueries) {
      ctx.reply(`🔎 Searching for: "${searchQuery}"`);
      
      try {
        const results = await searchNews(searchQuery);
        
        if (results && results.length > 0) {
          allResults = [...allResults, ...results];
        }
      } catch (error) {
        console.error(`Error searching for "${searchQuery}":`, error);
        ctx.reply(`Error searching for "${searchQuery}". Trying next query...`);
      }
    }
    
    // Remove duplicates based on URL
    allResults = allResults.filter((result, index, self) =>
      index === self.findIndex((r) => r.url === result.url)
    );
    
    if (allResults.length === 0) {
      return ctx.reply('No search results found. Try a different search term.');
    }
    
    ctx.reply(`Found ${allResults.length} initial results. Analyzing for freshness and relevance...`);
    
    // Process results in batches to save API calls
    const validResults = await processNewsResults(allResults);
    
    if (validResults.length === 0) {
      return ctx.reply('No breaking news stories found that match your criteria. Try a different search term.');
    }
    
    // Sort by score descending
    validResults.sort((a, b) => b.score - a.score);
    
    // Send results to Telegram
    await ctx.reply(`🗞️ Found ${validResults.length} breaking news stories:`);
    
    for (const result of validResults) {
      const message = `
📰 *${result.title}*
${result.snippet}

⏰ Published: ${result.publishedTime || 'Recently'}
🔗 [Read full article](${result.url})
🏆 Score: ${result.score.toFixed(1)}/10
${result.hasTikTok ? '📱 Mentions TikTok!' : ''}
`;
      
      await ctx.replyWithMarkdownV2(
        message
          .replace(/\./g, '\\.')
          .replace(/\!/g, '\\!')
          .replace(/\-/g, '\\-')
          .replace(/\#/g, '\\#')
          .replace(/\(/g, '\\(')
          .replace(/\)/g, '\\)')
      );
    }
    
  } catch (error) {
    console.error('Error during search:', error);
    ctx.reply('An error occurred during the search. Please try again later.');
  }
});

/**
 * Generates multiple search queries based on the original query
 */
async function generateSearchQueries(originalQuery) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: "You are a news search assistant. Generate 3 natural, colloquial search queries for breaking news based on the user's intent. Focus on finding specific events and named individuals/entities. Format as a JSON array with just the queries."
        },
        {
          role: "user",
          content: `Create search queries for breaking news about: ${originalQuery}. Include time-related terms like "today", "breaking", "just happened", "hours ago", etc.`
        }
      ],
      temperature: 0.7,
      max_tokens: 150,
      response_format: { type: "json_object" }
    });
    
    const content = response.choices[0].message.content;
    const parsedContent = JSON.parse(content);
    
    if (Array.isArray(parsedContent.queries)) {
      return [originalQuery, ...parsedContent.queries];
    } else {
      return [originalQuery];
    }
    
  } catch (error) {
    console.error('Error generating search queries:', error);
    return [originalQuery];
  }
}

/**
 * Processes news results to filter by time and relevance
 */
async function processNewsResults(results) {
  const validResults = [];
  const batchSize = 5; // Process 5 articles at a time to minimize API calls
  
  for (let i = 0; i < results.length; i += batchSize) {
    const batch = results.slice(i, i + batchSize);
    const batchPromises = batch.map(async (result) => {
      try {
        // Check if the article is fresh enough (within 8 hours)
        const isFresh = await isArticleFresh(result.url);
        
        if (!isFresh) {
          return null;
        }
        
        // Get article content for evaluation
        const content = await fetchArticleContent(result.url);
        
        if (!content) {
          return null;
        }
        
        // Check if content mentions TikTok
        const hasTikTok = content.toLowerCase().includes('tiktok');
        
        // Evaluate article with OpenAI
        const evaluation = await evaluateArticle(result.title, content);
        
        if (evaluation.isValid && evaluation.score >= 5) {
          return {
            ...result,
            score: hasTikTok ? evaluation.score + 0.5 : evaluation.score,
            hasTikTok,
            publishedTime: evaluation.publishedTime || 'Recently'
          };
        }
        
        return null;
      } catch (error) {
        console.error(`Error processing article ${result.url}:`, error);
        return null;
      }
    });
    
    const batchResults = await Promise.all(batchPromises);
    validResults.push(...batchResults.filter(result => result !== null));
  }
  
  return validResults;
}

/**
 * Checks if an article was published within the time threshold
 */
async function isArticleFresh(url) {
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      timeout: 5000
    });
    
    const $ = cheerio.load(response.data);
    
    // Try common meta tags for published time
    const possibleMetaTags = [
      'meta[property="article:published_time"]',
      'meta[name="pubdate"]',
      'meta[name="publishdate"]',
      'meta[name="date"]',
      'meta[itemprop="datePublished"]',
      'time[datetime]'
    ];
    
    for (const tag of possibleMetaTags) {
      const element = $(tag);
      if (element.length > 0) {
        const dateStr = element.attr('content') || element.attr('datetime');
        if (dateStr) {
          const publishedDate = new Date(dateStr);
          const now = new Date();
          const timeDiff = now - publishedDate;
          
          return !isNaN(publishedDate) && timeDiff <= TIME_THRESHOLD;
        }
      }
    }
    
    // If no date found, check for common date patterns in the text
    const bodyText = $('body').text();
    const dateRegexPatterns = [
      /(\d{1,2})\s+(hour|hours|hr|hrs)\s+ago/i,
      /(\d{1,2}):(\d{2})\s+(am|pm)\s+today/i,
      /published\s+(\d{1,2})\s+(hour|hours|hr|hrs)\s+ago/i
    ];
    
    for (const pattern of dateRegexPatterns) {
      const match = bodyText.match(pattern);
      if (match) {
        // If we find hours ago pattern, check if it's within threshold
        if (match[1] && parseInt(match[1]) <= 8) {
          return true;
        }
      }
    }
    
    // If we can't determine time, default to checking
    // if "hours ago", "minutes ago", or today's date appears
    const todayDate = moment().format('MMMM D, YYYY');
    const yesterdayDate = moment().subtract(1, 'day').format('MMMM D, YYYY');
    
    return (
      bodyText.includes('hours ago') ||
      bodyText.includes('hour ago') ||
      bodyText.includes('minutes ago') ||
      bodyText.includes('minute ago') ||
      bodyText.includes('just now') ||
      bodyText.includes('breaking') ||
      bodyText.includes(todayDate) ||
      bodyText.includes(yesterdayDate.substring(0, 10))
    );
    
  } catch (error) {
    console.error(`Error checking if article is fresh: ${url}`, error);
    return false;
  }
}

/**
 * Fetches article content for evaluation
 */
async function fetchArticleContent(url) {
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      timeout: 5000
    });
    
    const $ = cheerio.load(response.data);
    
    // Remove script and style elements
    $('script, style').remove();
    
    // Try to get content from article tags first
    let content = $('article').text();
    
    // If no article tag, try common content containers
    if (!content || content.length < 100) {
      content = $('.article-body, .article-content, .story-body, .story-content, .news-content, .post-content').text();
    }
    
    // If still no content, get body text but limit it
    if (!content || content.length < 100) {
      content = $('body').text();
    }
    
    // Clean and normalize whitespace
    content = content
      .replace(/\s+/g, ' ')
      .trim()
      .substring(0, 5000); // Limit to 5000 chars to save tokens
    
    return content;
  } catch (error) {
    console.error(`Error fetching article content: ${url}`, error);
    return null;
  }
}

/**
 * Evaluates article content using OpenAI
 */
async function evaluateArticle(title, content) {
  // Extract a snippet for evaluation
  const snippet = content.substring(0, 3000); // Limited to save tokens
  
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `You evaluate news articles based on specific criteria. Determine if an article meets these criteria:
1) Is breaking news about a named individual/entity and specific event
2) Is a new unique/unusual story about a specific thing/event
3) Is an evolving story that could have future updates

Avoid:
- Opinion pieces
- List articles
- General articles about trends (not specific events)

Respond in JSON format with these fields:
- isValid: boolean (true if it meets criteria)
- score: number (1-10, how well it matches criteria)
- reason: short explanation
- publishedTime: extract when article was published if mentioned
- category: which criteria it fits (breaking, unique, evolving)`
        },
        {
          role: "user",
          content: `Article Title: ${title}\n\nArticle Content Snippet: ${snippet}`
        }
      ],
      temperature: 0.3,
      max_tokens: 150,
      response_format: { type: "json_object" }
    });
    
    const content = response.choices[0].message.content;
    return JSON.parse(content);
    
  } catch (error) {
    console.error('Error evaluating article with OpenAI:', error);
    return {
      isValid: false,
      score: 0,
      reason: 'Error during evaluation'
    };
  }
}

// Start the bot
bot.launch().then(() => {
  console.log('Breaking News Bot is running!');
}).catch(err => {
  console.error('Error starting bot:', err);
});

// Enable graceful stop
process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));